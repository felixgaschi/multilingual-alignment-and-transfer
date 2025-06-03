"""
Script to compare different realignment methods with simple fine-tuning
"""


import os
import sys
import logging
import datasets
from typing import List
from contextlib import ExitStack
from transformers import AutoTokenizer, set_seed

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
from multilingual_eval.training.evaluation_loops import evaluate_xquad


def train(
    left_lang: str,
    right_langs: List[str],
    eval_langs: List[str],
    translation_dir: str,
    fastalign_dir: str,
    dico_dir: str,
    awesome_dir: str,
    config=None,
    sweep_config=None,
    zh_segmenter=None,
    debug=False,
    cache_dir=None,
    large_gpu=False,
    n_epochs=5,
    layers=None,
    result_store=None,
    additional_realignment_langs=None,
    realignment_steps=None,
    checkpoint_path=None,
    no_interleave=True,
):
    layers = layers or [-1]
    model_name = config["model"]
    task_name = config["task"]
    seed = config["seed"]
    method = config["method"]
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
    cumul_batch_size = 32
    batch_size = get_batch_size(model_name, cumul_batch_size, large_gpu=large_gpu)
    accumulation_steps = cumul_batch_size // batch_size

    realignment_batch_size = 16

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
    set_seed(seed)
    if method == "baseline":
        model = model_fn(task_name, with_realignment=False)(
            model_name, cache_dir=model_cache_dir
        )
    else:
        model = model_fn(task_name, with_realignment=True)(
            model_name,
            cache_dir=model_cache_dir,
            nb_pairs=len(right_langs),
            strong_alignment=True,
            realignment_loss="contrastive",
            with_mapping=False,
            regularization_to_init=False,
            realignment_layers=layers,
        )

    logging.debug(model)

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
    lang_pairs = [(left_lang, right_lang) for right_lang in right_langs + (additional_realignment_langs or [])]
    if aligner == "fastalign":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            fastalign_dir,
            lang_pairs,
            max_length=96,
            seed=seed,
        )
    elif aligner == "dico":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer, translation_dir, dico_dir, lang_pairs, max_length=96, seed=seed, do_interleave_datasets=(not no_interleave)
        )
    elif aligner == "awesome":
        alignment_dataset = get_multilingual_realignment_dataset(
            tokenizer,
            translation_dir,
            awesome_dir,
            lang_pairs,
            max_length=96,
            seed=seed,
        )
    elif aligner is None:
        alignment_dataset = None
    else:
        raise KeyError(aligner)

    # perform realignment and fine-tuning
    realignment_training_loop(
        tokenizer,
        model,
        training_dataset,
        alignment_dataset,
        strategy=method,
        evaluation_datasets=validation_datasets if task_name not in ["xquad"] else None,
        same_language_evaluation_dataset=source_validation_dataset
        if task_name not in ["xquad"]
        else None,
        evaluation_prefixes=eval_langs,
        seed=seed,
        task_batch_size=batch_size,
        learning_rate=7.5e-6 if "roberta" in model_name else 2e-5,
        realignment_batch_size=realignment_batch_size,
        realignment_steps_by_finetuning=1,
        n_epochs=n_epochs,
        accumulation_steps=accumulation_steps,
        result_store=result_store,
        metric_fn=get_dataset_metric_fn(task_name)(),
        data_collator=collator_fn(task_name)(tokenizer),
        model_name=model_name,
        nb_realignment_steps_before=realignment_steps,
        checkpoint_path=checkpoint_path,
    )

    if task_name == "xquad":
        evaluate_xquad(
            model,
            tokenizer,
            left_lang,
            right_langs,
            batch_size=batch_size,
            debug=False,
            data_cache_dir=data_cache_dir,
            log_in_wandb=False,
            result_store=result_store
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    default_strategies = [
        "baseline",
        *[
            f"{strategy}_{aligner}"
            for strategy in ["during", "before"]
            for aligner in ["fastalign", "dico", "awesome"]
        ],
    ]

    parser = argparse.ArgumentParser()
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
        "--strategies",
        nargs="+",
        type=str,
        default=default_strategies,
        help="Realignment strategies to use, of the form strategy_aligner, with strategy being either before or after and aligner being either dico, fastalign or awesome",
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
        default=["ar", "es", "fr", "ru", "zh", "af", "fa", "hi"],
        help="Target languages for cross-lingual transfer",
    )
    parser.add_argument(
        "--eval_langs",
        type=str,
        nargs="+",
        default=None,
        help="Target eval languages for cross-lingual transfer",
    )
    parser.add_argument(
        "--additional_realignment_langs",
        type=str,
        nargs="+",
        default=None,
        help="Target languages for cross-lingual transfer",
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
    parser.add_argument("--project_name", type=str, default="")
    parser.set_defaults(debug=False, large_gpu=False, use_wandb=False, no_interleave=False)
    args = parser.parse_args()
    
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

    # Config with all the different values of run parameters
    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": seeds[: args.n_seeds] if args.seeds is None else args.seeds},
            "model": {"values": args.models},
            "task": {"values": args.tasks},
            "method": {"values": args.strategies},
        },
    }

    if args.debug:
        sweep_config["parameters"]["seed"]["values"] = sweep_config["parameters"][
            "seed"
        ]["values"][:1]

    with ExitStack() as stack:
        #Currently, NLI does not need zh_segmenter
        # if "zh" in args.right_langs + (args.additional_realignment_langs or []) or args.left_lang == "zh":
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
                    layers=args.layers,
                    config=cfg,
                    sweep_config=sweep_cfg,
                    zh_segmenter=zh_sgm,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                    additional_realignment_langs=args.additional_realignment_langs,
                    realignment_steps=args.realignment_steps,
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
                    layers=args.layers,
                    config=run_config,
                    sweep_config=sweep_config,
                    zh_segmenter=zh_segmenter,
                    debug=args.debug,
                    large_gpu=args.large_gpu,
                    cache_dir=args.cache_dir,
                    n_epochs=args.n_epochs,
                    result_store=result_store,
                    additional_realignment_langs=args.additional_realignment_langs,
                    realignment_steps=args.realignment_steps,
                    checkpoint_path=args.checkpoint_path,
                    no_interleave=args.no_interleave
                )
                recorder.add(result_store.get_results())
