"""
Script to compare different realignment methods with simple fine-tuning
"""


import os
import sys
import logging
import datasets
from typing import List, Optional
from contextlib import ExitStack
from transformers import AutoTokenizer, set_seed
import numpy as np

from dataclasses import dataclass, field

sys.path.append(os.curdir)

from multilingual_eval.seeds import seeds
from multilingual_eval.loggers import (
    WandbResultStore,
    DictResultStore,
    DefaultResultStore,
)
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.training.wandb_utils import (
    wrap_train,
    imitate_wandb_sweep,
    store_dicts_in_csv,
    CSVRecorder,
)
from multilingual_eval.training.training_loops import realignment_training_loop
from multilingual_eval.training.batch_sizes import get_batch_size
from multilingual_eval.datasets.dispatch_datasets import (
    get_dataset_fn,
    get_dataset_metric_fn,
    model_fn,
    collator_fn,
)
from multilingual_eval.datasets.realignment_task import (
    get_multilingual_realignment_dataset,
)
from multilingual_eval.training.evaluation_loops import evaluate_xquad, evaluate_xlsum
from multilingual_eval.models.simplified import SimplifiedModelForRealignment


def train(
    left_lang: str,
    right_langs: List[str],
    eval_langs: List[str],
    translation_dir: str,
    fastalign_dir: str,
    dico_dir: str,
    awesome_dir: str,
    learning_rate=None,
    africlir_root_dir="../africlirmatrix/test",
    ### Parameters for weighted sampling
    enable_weighted_sampling=False,
    weighted_sampling_method="meta_learning",
    decouple_meta_and_model_updates="joint",
    meta_loss_type="micro",
    meta_learning_rate=1e-2,
    inner_batches_before_outer=10,
    lbsmoothing_eps=0,
    with_regularization=False,
    lambda_entropy=1e-3,
    noise_mixing_strat=None,
    softmax_temp=1,
    ucb_exploration_coef=1.0,
    static_meta_weights_jsonl=None,
    ###
    config=None,
    sweep_config=None,
    zh_segmenter=None,
    debug=False,
    cache_dir=None,
    large_gpu=False,
    n_epochs=5,
    layers=None,
    result_store=None,
    realignment_steps=None,
    realignment_batch_size=16,
    extra_realignment_steps_checkpoints=None,
    checkpoint_path=None,
    no_interleave=True,
    use_adapter=False,
    adapter_approach="same", # "same", "separate", or "realign_only"
    adapter_config=None,
):
    layers = layers or [-1]
    model_name = config["model"]
    task_name = config["task"]
    seed = config["seed"]
    method = config["method"]
    n_realignment_langs = config.get("n_realignment_langs", None)
    if "baseline" in method:
        aligner = None
    else:
        # method, aligner = method.split("_")
        aligner = method.split("_")[-1]
        method = "_".join(x for x in method.split("_")[:-1])
        
    print(f'METHOD : {method}')
    print(f'ALIGNER: {aligner}')

    # result_store allows to gather information along the experiment
    # By default, only logs them in the console
    result_store = result_store or DefaultResultStore()

    # Compute batch size and gradient accumulation from real batch size (32)
    # and an empirical batch size based on the model name
    target_models = ("gemma", "llama", "mt5-large")
    cumul_batch_size = 32 if any(m in model_name.lower() for m in target_models) else 128

    batch_size = get_batch_size(model_name, cumul_batch_size, large_gpu=large_gpu)
    accumulation_steps = cumul_batch_size // batch_size
    print(f"Finetuning batch size: {batch_size}, real batch size: {cumul_batch_size}, accumulation steps: {accumulation_steps}")

    # realignment_batch_size = 128
    # if "gemma-2-9b" in model_name:
    #     realignment_batch_size = 16
    print(f"Realignment batch size: {realignment_batch_size}")

    assert cumul_batch_size % batch_size == 0

    # Compute caching directory for HuggingFace datasets and models
    data_cache_dir = (
        os.path.join(cache_dir, "datasets") if cache_dir is not None else cache_dir
    )
    model_cache_dir = (
        os.path.join(cache_dir, "transformers") if cache_dir is not None else cache_dir
    )

    # Instantiate tokenizer, model and set the seed
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    if "meta-llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        
    set_seed(seed)

    if n_realignment_langs is None:
        n_realignment_langs = len(right_langs)
    if n_realignment_langs < len(right_langs):
        logging.info(
            f"Using only {n_realignment_langs} out of {len(right_langs)} target languages for realignment"
        )
        selected_langs = np.random.choice(right_langs, (n_realignment_langs,), replace=False)
        right_langs = selected_langs.tolist()
    else:
        selected_langs = right_langs
    result_store.log({"realignment_langs": ",".join(sorted(selected_langs))})

    if method == "baseline":
        model = model_fn(task_name, with_realignment=False, simplified=True, llama_qa_hotfix_dir=os.path.join(cache_dir, "tmp_llama"))(
            model_name, cache_dir=model_cache_dir
        )
    else:
        model = model_fn(task_name, with_realignment=True, simplified=True, llama_qa_hotfix_dir=os.path.join(cache_dir, "tmp_llama"))(
            model_name,
            cache_dir=model_cache_dir,
            nb_pairs=len(selected_langs),
            strong_alignment=True,
            realignment_loss="contrastive",
            realignment_method="sentence" if aligner == "noaligner" else "token",
            with_mapping=False,
            regularization_to_init=False,
            realignment_layers=layers,
        )

    if "meta-llama" in model_name:
        model_to_adapt = model.model if isinstance(model, SimplifiedModelForRealignment) else model
        model_to_adapt.config.pad_token_id = model_to_adapt.config.eos_token_id

    always_frozen_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            always_frozen_params.append(name)

    if use_adapter and (method != "baseline" or adapter_approach == "same"):
        # from peft import LoraConfig, get_peft_model
        from peft import get_peft_model

        # adapter_config = LoraConfig(
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        if not adapter_config:
            raise Exception("Adapter LoRA config is expected when use_adapter is set to True.")
        if isinstance(model, SimplifiedModelForRealignment):
            model.model = get_peft_model(model.model, adapter_config, adapter_name="main_adapter")
        else:
            model = get_peft_model(model, adapter_config, adapter_name="main_adapter")
        model_to_adapt = model.model if isinstance(model, SimplifiedModelForRealignment) else model

        model_to_adapt.set_adapter("main_adapter")

    logging.info(f"Model use adapter: {use_adapter} with approach: {adapter_approach}")

    logging.info(model)

    if "clirmatrix" in task_name:
        training_dataset, validation_datasets, source_validation_dataset = None, None, None
    else:
        # Load fine-tuning dataset
        training_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
            left_lang,
            tokenizer,
            split="train",
            limit=1000 if debug else None,
            datasets_cache_dir=data_cache_dir,
            max_length=96
        )

        # Load test dataset for target languages
        validation_datasets = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
            eval_langs,
            tokenizer,
            split="test",
            limit=100 if debug else None,
            datasets_cache_dir=data_cache_dir,
            interleave=False,
        )

        # Load test dataset for source language
        source_validation_dataset = get_dataset_fn(task_name, zh_segmenter=zh_segmenter)(
            left_lang,
            tokenizer,
            split="test",
            limit=100 if debug else None,
            datasets_cache_dir=data_cache_dir,
        )

        print(f"training dataset: {len(training_dataset)}")
        for lang, ds in zip(eval_langs, validation_datasets):
            print(f"{lang} dataset: {len(ds)}")

    # Load realignment datatset
    lang_pairs = [(left_lang, right_lang) for right_lang in selected_langs]
    if enable_weighted_sampling:
        logging.info(f"Weighted sampling is enable. Alignment dataset will be load seperately as an iterator for each language. Meta loss type: {meta_loss_type}.")
    if aligner == "fastalign":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            fastalign_dir,
            lang_pairs,
            max_length=96,
            seed=seed,
            return_iterator_list=enable_weighted_sampling,
        )
    elif aligner == "dico":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, dico_dir, lang_pairs, max_length=96, seed=seed, do_interleave_datasets=(not no_interleave), return_iterator_list=enable_weighted_sampling,
        )
    elif aligner == "awesome":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            awesome_dir,
            lang_pairs,
            max_length=96,
            seed=seed,
            return_iterator_list=enable_weighted_sampling,
        )
    elif aligner == "noaligner":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            None,
            lang_pairs,
            max_length=96,
            seed=seed,
            first_subowrd_only=True, # Maybe True is still better because we don't have issues with re-weighting languages with high fertility
            return_iterator_list=enable_weighted_sampling,
        )
    elif aligner is None:
        alignment_dataset = None
    else:
        raise KeyError(aligner)

    if not learning_rate:
        if use_adapter:
            learning_rate = 1e-4
        else:
            learning_rate = 7.5e-6 if "roberta" in model_name else 2e-5
    logging.info(f"Learning rate used: {learning_rate}")        
    # perform realignment and fine-tuning
    realignment_training_loop(
        tokenizer,
        model,
        training_dataset,
        alignment_dataset,
        strategy=method,
        evaluation_datasets=validation_datasets
        if task_name not in ["xquad", "xlsum", "xlsum_clm"]
        else None,
        same_language_evaluation_dataset=source_validation_dataset
        if task_name not in ["xquad", "xlsum", "xlsum_clm"]
        else None,
        evaluation_prefixes=eval_langs,
        seed=seed,
        task_batch_size=batch_size,
        learning_rate=learning_rate,
        realignment_batch_size=realignment_batch_size,
        realignment_steps_by_finetuning=1,
        n_epochs=n_epochs,
        accumulation_steps=accumulation_steps,
        ### Parameters for weighted sampling
        enable_weighted_sampling=enable_weighted_sampling,
        weighted_sampling_method=weighted_sampling_method,
        decouple_meta_and_model_updates=decouple_meta_and_model_updates,
        meta_loss_type=meta_loss_type,
        meta_learning_rate=meta_learning_rate,
        inner_batches_before_outer=inner_batches_before_outer,
        lbsmoothing_eps=lbsmoothing_eps,
        with_regularization=with_regularization,
        lambda_entropy=lambda_entropy,
        noise_mixing_strat=noise_mixing_strat,
        softmax_temp=softmax_temp,
        ucb_exploration_coef=ucb_exploration_coef,
        static_meta_weights_jsonl=static_meta_weights_jsonl,
        ###
        result_store=result_store,
        metric_fn=get_dataset_metric_fn(task_name)() if "clirmatrix" not in task_name else None,
        data_collator=collator_fn(task_name)(tokenizer) if "clirmatrix" not in task_name else None,
        model_name=model_name,
        nb_realignment_steps_before=realignment_steps,
        extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
        checkpoint_path=checkpoint_path,
        task_name=task_name,
        noaligner=(aligner == "noaligner"),
        use_adapter=use_adapter,
        adapter_approach=adapter_approach,
        adapter_config=adapter_config,
        always_frozen_params=always_frozen_params,
    )

    if "clirmatrix" in task_name:
        from transformers import PreTrainedModel
        from multilingual_eval.models.stwrapper import build_sentence_transformer
        from multilingual_eval.information_retrieval.eval_clirmatrix import run_clir_eval_many
        from multilingual_eval.information_retrieval.train_msmacro import train_clir

        os.environ['IR_DATASETS_HOME'] = data_cache_dir

        transformer_model = getattr(model, "model", model)
        transformer_model = getattr(transformer_model, "base_model", transformer_model)
        if not isinstance(transformer_model, PreTrainedModel):
            raise TypeError(f"Model type {type(model).__name__} not supported for clirmatrix task")
        sentence_model = build_sentence_transformer(transformer_model, tokenizer, normalize=True)

        if task_name != "clirmatrix_noft":
            print('\nSTARTING FINETUNING CLIR\n')
            train_clir(
                sentence_model, 
                datasets_cache_dir=data_cache_dir, 
                per_device_train_batch_size=16 if "gemma" in model_name else 64,
                )
            print('\nDONE FINETUNING\n')

        results = run_clir_eval_many(
            model=sentence_model,
            doc_langs=eval_langs,
            africlir_root_dir=africlir_root_dir,
            batch_size=16 if "gemma" in model_name else 64,
            debug=True,
            debug_num_queries=300,
            datasets_cache_dir=data_cache_dir,
        )
        logging.info(results)
        result_store.log(results)


    if task_name == "xquad":
        evaluate_xquad(
            model,
            tokenizer,
            left_lang,
            eval_langs,
            batch_size=batch_size,
            debug=False,
            data_cache_dir=data_cache_dir,
            log_in_wandb=False,
            result_store=result_store
    )

    if task_name in ["xlsum", "xlsum_clm"]:
        evaluate_xlsum(
            model,
            tokenizer,
            left_lang,
            eval_langs,
            batch_size=batch_size,
            debug=debug,
            data_cache_dir=data_cache_dir,
            log_in_wandb=False,
            result_store=result_store,
        )


@dataclass
class LoraArguments:
    """
    Arguments tracking the adapter configurations.
    """
    lora_r: int = field(
        default=8, 
        metadata={"help": "The intrinsic rank of the LoRA delta matrix updates."}
    )
    lora_alpha: int = field(
        default=32, 
        metadata={"help": "The scaling hyperparameter for the LoRA weight matrices."}
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules to target. Pass multiples like: --target_modules q_proj v_proj"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules to fully train (e.g., --modules_to_save embed_tokens score)."}
    )
    lora_dropout: float = field(
        default=0.1, 
        metadata={"help": "The dropout probability applied to the LoRA layers."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'."}
    )

if __name__ == "__main__":
    # import argparse
    from transformers import HfArgumentParser


    logging.basicConfig(level=logging.INFO)

    default_strategies = [
        "baseline",
        *[
            f"{strategy}_{aligner}"
            for strategy in ["during", "before"]
            for aligner in ["fastalign", "dico", "awesome", "noaligner"]
        ],
    ]

    # parser = argparse.ArgumentParser()
    parser = HfArgumentParser(LoraArguments)
    parser.add_argument(
        "--translation_dir",
        type=str,
        default=None,
        help="Directory where the parallel dataset can be found, must be set if other strategy than baseline is used.",
    )
    parser.add_argument(
        "--fastalign_dir",
        type=str,
        default=None,
        help="Directory where fastalign alignments can be found, must be set if strategy ending in _fastalign is used",
    )
    parser.add_argument(
        "--dico_dir",
        type=str,
        help="Directory where bilingual dictionary alignments can be found, must be set if strategy ending in _dico is used",
    )
    parser.add_argument(
        "--awesome_dir",
        type=str,
        help="Directory where awesome alignments can be found, must be set if strategy ending in awesome is used",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=[
            "bert-base-multilingual-cased",
            "xlm-roberta-base",
            "distilbert-base-multilingual-cased",
        ],
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, default=["wikiann", "udpos", "xnli", "xquad"]
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate used for both realignment and fine-tuning.",
    )
    parser.add_argument(
        "--africlir_root_dir",
        type=str,
        default='../africlirmatrix/test',
        help="Directory path for AfriCLIRmatrix test files (default: ../africlirmatrix/test), must clone the repo first",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        type=str,
        default=default_strategies,
        help="Realignment strategies to use, of the form strategy_aligner, with strategy being either before or after and aligner being either dico, fastalign or awesome",
    )
    parser.add_argument(
        "--enable_weighted_sampling",
        action="store_true",
        dest="enable_weighted_sampling",
        help="Switch from random uniform sampling to weighted sampling with learnable probability",
    )
    parser.add_argument("--meta_loss_type", type=str, default="micro", choices=["micro", "macro", "micro_log"])
    parser.add_argument(
        "--weighted_sampling_method",
        type=str,
        default="meta_learning",
        choices=["meta_learning", "ucb"],
        help="Method for learning language sampling weights: 'meta_learning' (gradient-based) or 'ucb' (multi-armed bandit).",
    )
    parser.add_argument(
        "--decouple_meta_and_model_updates",
        type=str,
        default="joint",
        choices=["joint", "decoupled", "decoupled_closed"],
        dest="decouple_meta_and_model_updates",
        help=(
            "For meta_learning weighted sampling only. "
            "'joint': learn meta-weights and model jointly. "
            "'decoupled': two phases — (1) learn meta-weights with model frozen "
            "via gradient ascent, then (2) freeze meta-weights and train model. "
            "'decoupled_closed': same two-phase structure but phase (1) uses a "
            "closed-form estimate (softmax of average per-language losses, no "
            "backprop)."
        ),
    )
    parser.add_argument(
        "--ucb_exploration_coef",
        type=float,
        default=1.0,
        help="Exploration coefficient 'c' for UCB formula: UCB_i = mean_reward_i + c * sqrt(ln(t) / (n_i + 1)). Higher values favor exploration.",
    )
    parser.add_argument(
        "--static_meta_weights_jsonl",
        type=str,
        default=None,
        help=(
            "Path to a meta_weights_overtime.jsonl file. When used with "
            "decoupled weighted sampling, phase 1 is skipped and the last "
            "saved meta_weight is used for static phase-2 realignment."
        ),
    )
    parser.add_argument(
        "--left_lang",
        type=str,
        default="en",
        help="Source language for cross-lingual transfer",
    )
    parser.add_argument(
        "--right_langs",
        type=str,
        nargs="+",
        default=None,
        help="Target languages for cross-lingual transfer",
    )
    parser.add_argument(
        "--inner_batches_before_outer",
        type=int,
        default=5,
        help="The number of inner loops to run before the outer loop is optimized.",
    )
    parser.add_argument(
        "--meta_learning_rate",
        type=float,
        default=1e-2,
        help="The learning rate for the outer loop optimization.",
    )
    parser.add_argument(
        "--lbsmoothing_eps",
        type=float,
        default=0,
        help="Epsilon to smooth the sampling probs with uniform distribution by epsilon value percentage.",
    )
    parser.add_argument(
        "--with_regularization",
        action="store_true",
        dest="with_regularization",
        help="Option to whether include entropy regularization to the outer loop optimization.",
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=1e-3,
        help="Hyperparameter to modify the contribution of entropy regularization to the outer loop loss.",
    )
    parser.add_argument(
        "--noise_mixing_strat", 
        type=str, 
        help=(
            "Strategy to mix noises and amount (>0). "
            "Format: {strat}_{noise_type}_{amount} (e.g., 'examples_reverse_10'). "
            "Valid strats: ['examples', 'batches']. "
            "Valid noises: ['uniform', 'reverse']."
        ),
        default=None,
    )
    parser.add_argument(
        "--softmax_temp", 
        type=float,
        help="Temperature to smooth softmax distribution, specifically softmax(weights/temp).",
        default=1,
    )
    parser.add_argument(
        "--n_realignment_langs",
        type=int,
        nargs="+",
        default=None,
        help="Number of target languages to use for realignment, if None, all target languages are used (default: None)",
    )
    parser.add_argument(
        "--eval_langs",
        type=str,
        nargs="+",
        default=None,
        help="Target eval languages for cross-lingual transfer",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory which will contain subdirectories 'transformers' and 'datasets' for caching HuggingFace models and datasets",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="If using wandb, useful to restart a sweep or launch several run in parallel for a same sweep",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="Use this to perform a quicker test run with less samples",
    )
    parser.add_argument(
        "--large_gpu",
        action="store_true",
        dest="large_gpu",
        help="Use this option for 45GB GPUs (less gradient accumulation needed)",
    )
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[-1],
        help="The layer (or list of layers) on which we want to perform realignment (default -1 for the last one)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The path to the output CSV file containing results (used only if wandb is not use, which is the case by default)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="The path to checkpoint save directory",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        dest="use_wandb",
        help="Use this option to use wandb (but must be installed first)",
    )
    parser.add_argument(
        "--no_interleave",
        action="store_true",
        dest="no_interleave",
        help="Use this option to disable interleave",
    )
    parser.add_argument(
        "--segmenter_port",
        type=int,
        default=9123
    )
    parser.add_argument(
        "--realignment_steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--realignment_batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--extra_realignment_steps_checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="The extra realignment checkpoints to save along the realignment process (Must be < than realignment_steps).",
    )
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--use_adapter", action="store_true", dest="use_adapter")
    parser.add_argument("--adapter_approach", type=str, default="same", choices=["same", "separate", "realign_only"])
    parser.set_defaults(debug=False, large_gpu=False, use_wandb=False, no_interleave=False, use_adapter=False)
    args = parser.parse_args()

    if args.use_adapter:
        from peft import LoraConfig

        adapter_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            modules_to_save=args.modules_to_save,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        print(f"LoRA Configuration successfully built: {adapter_config}")
    else:
        adapter_config = None
    
    if not args.eval_langs:
        args.eval_langs = args.right_langs

    if not args.use_wandb and args.output_file is None:
        raise Exception(
            f"Either wandb must be used (--use_wandb) or an output csv file must be set (--output_file) to store results"
        )
    if not args.use_wandb:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.checkpoint_path:
        os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.n_realignment_langs is None:
        n_realignment_langs = [len(args.right_langs)]
    else:
        n_realignment_langs = args.n_realignment_langs

    # Config with all the different values of run parameters
    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": seeds[: args.n_seeds] if args.seeds is None else args.seeds},
            "model": {"values": args.models},
            "task": {"values": args.tasks},
            "method": {"values": args.strategies},
            "n_realignment_langs": {
                "values": n_realignment_langs
            },
        },
    }

    if args.debug:
        sweep_config["parameters"]["seed"]["values"] = sweep_config["parameters"][
            "seed"
        ]["values"][:1]

    with ExitStack() as stack:
        #Currently, NLI does not need zh_segmenter
        # if "zh" in args.right_langs or args.left_lang == "zh":
        #     # Calls Stanford Segmenter in another process, hence the context manager
        #     zh_segmenter = stack.enter_context(StanfordSegmenter(port=args.segmenter_port))
        # else:
        #     zh_segmenter = None
        zh_segmenter = None
        
        if args.output_file:
            recorder = stack.enter_context(CSVRecorder(args.output_file, config_props=list(sweep_config["parameters"].keys())))
        else:
            recorder = None

        if args.use_wandb:
            import wandb

            result_store = WandbResultStore()

            if args.sweep_id is None:
                sweep_id = wandb.sweep(sweep_config, project=args.project_name or None)
            else:
                sweep_id = args.sweep_id

            final_train_fn = wrap_train(
                lambda cfg, sweep_cfg, zh_sgm: train(
                    args.left_lang,
                    args.right_langs,
                    args.eval_langs,
                    args.translation_dir,
                    args.fastalign_dir,
                    args.dico_dir,
                    args.awesome_dir,
                    learning_rate=args.learning_rate,
                    africlir_root_dir=args.africlir_root_dir,
                    ### Parameters for weighted sampling
                    enable_weighted_sampling=args.enable_weighted_sampling,
                    weighted_sampling_method=args.weighted_sampling_method,
                    decouple_meta_and_model_updates=args.decouple_meta_and_model_updates,
                    meta_loss_type=args.meta_loss_type,
                    meta_learning_rate=args.meta_learning_rate,
                    inner_batches_before_outer=args.inner_batches_before_outer,
                    lbsmoothing_eps = args.lbsmoothing_eps,
                    with_regularization=args.with_regularization,
                    lambda_entropy=args.lambda_entropy,
                    noise_mixing_strat=args.noise_mixing_strat,
                    softmax_temp=args.softmax_temp,
                    ucb_exploration_coef=args.ucb_exploration_coef,
                    static_meta_weights_jsonl=args.static_meta_weights_jsonl,
                    ###
                    layers=args.layers,
                    config=cfg,
                    sweep_config=sweep_cfg,
                    zh_segmenter=zh_sgm,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                    realignment_steps=args.realignment_steps,
                    realignment_batch_size=args.realignment_batch_size,
                    extra_realignment_steps_checkpoints=args.extra_realignment_steps_checkpoints,
                    use_adapter=args.use_adapter,
                    adapter_approach=args.adapter_approach,
                    adapter_config=adapter_config,
                ),
                sweep_config,
                sweep_id,
                zh_segmenter=zh_segmenter,
            )

            wandb.agent(sweep_id, final_train_fn, project=args.project_name or None)
        else:
            datasets.disable_progress_bar()
            results = []
            # Looping over all possible configuration of runs provided in sweep_config
            for run_config in imitate_wandb_sweep(sweep_config):
                result_store = DictResultStore()
                result_store.log(run_config)
                if recorder.is_already_passed(run_config):
                    logging.info("This config was already run. Will ignore it")
                    continue
                train(
                    args.left_lang,
                    args.right_langs,
                    args.eval_langs,
                    args.translation_dir,
                    args.fastalign_dir,
                    args.dico_dir,
                    args.awesome_dir,
                    learning_rate=args.learning_rate,
                    africlir_root_dir=args.africlir_root_dir,
                    ### Parameters for weighted sampling
                    enable_weighted_sampling=args.enable_weighted_sampling,
                    weighted_sampling_method=args.weighted_sampling_method,
                    decouple_meta_and_model_updates=args.decouple_meta_and_model_updates,
                    meta_loss_type=args.meta_loss_type,
                    meta_learning_rate=args.meta_learning_rate,
                    inner_batches_before_outer=args.inner_batches_before_outer,
                    lbsmoothing_eps=args.lbsmoothing_eps,
                    with_regularization=args.with_regularization,
                    lambda_entropy=args.lambda_entropy,
                    noise_mixing_strat=args.noise_mixing_strat,
                    softmax_temp=args.softmax_temp,
                    ucb_exploration_coef=args.ucb_exploration_coef,
                    static_meta_weights_jsonl=args.static_meta_weights_jsonl,
                    ###
                    layers=args.layers,
                    config=run_config,
                    sweep_config=sweep_config,
                    zh_segmenter=zh_segmenter,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                    realignment_steps=args.realignment_steps,
                    realignment_batch_size=args.realignment_batch_size,
                    extra_realignment_steps_checkpoints=args.extra_realignment_steps_checkpoints,
                    checkpoint_path=args.checkpoint_path,
                    no_interleave=args.no_interleave,
                    use_adapter=args.use_adapter,
                    adapter_approach=args.adapter_approach,
                    adapter_config=adapter_config,
                )
                recorder.add(result_store.get_results())
