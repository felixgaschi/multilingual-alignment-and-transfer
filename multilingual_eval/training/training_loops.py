import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import get_scheduler
from torch.optim import Adam
import numpy as np
import random
import math
import hashlib
import os
import glob
import re
import json
import dataclasses

from multilingual_eval.training.states import TrainingState
from multilingual_eval.training.epoch_loop import epoch_loop
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator,
)
from multilingual_eval.training.evaluation_loops import (
    evaluate_several_token_classification,
    evaluate_token_classification,
)

def select_layers_for_realignment(total_layers, K):
    """
    Selects K layers by dividing the model into K blocks and choosing one layer per block.

    Args:
        total_layers (int): Total number of layers in the model.
        K (int): Number of layers to select.

    Returns:
        list: Selected layer indices (sorted).
    """
    if K > total_layers:
        return list(range(total_layers))

    # Divide layers into K blocks
    block_size = total_layers // K
    blocks = []
    for i in range(K):
        start = i * block_size
        end = (i + 1) * block_size if i < K - 1 else total_layers  # Last block takes remaining layers
        blocks.append((start, end))

    # Select one layer from each block
    selected_layers = []
    for block in blocks:
        start, end = block
        if len(selected_layers) > 0 and selected_layers[-1] == start - 1:
            start += 1
        selected_layers.append(random.choice(range(start, end)))
    
    return sorted(selected_layers)

def get_high_ani_layers_for_realignment(model_name: str, method: str):
    """
    model_name: ["xlm-roberta-base", "bert-base-multilingual-cased", "distilbert-base-multilingual-cased"]
    method: ["mean", "half"]
    """
    import pandas as pd
    ani_pretrain_res = pd.read_csv("scripts/2025_alignfreeze_continuation/distillation/anisotropy_results/pretrain/anisotropy_pretrain_1000.csv")[lambda x: x['Model'] == model_name]
    if method == "mean":
        mean_ani = ani_pretrain_res.Avg_anisotropy_xlang.mean()
        high_ani_layers = ani_pretrain_res[lambda x: x['Avg_anisotropy_xlang'] > mean_ani].Layer.tolist()
    elif method in ["half", "onethird"]:
        ani_pretrain_res_sorted = ani_pretrain_res.sort_values(by='Avg_anisotropy_xlang', ascending=False)
        if method == "half":
            num_layers_to_return = len(ani_pretrain_res_sorted) // 2 
        else:
            num_layers_to_return = len(ani_pretrain_res_sorted) // 3 
        high_ani_layers = ani_pretrain_res_sorted['Layer'].head(num_layers_to_return).tolist()
    else:
        raise NotImplementedError(f"Method {method} is not implemented for model {model_name}")
    return sorted(high_ani_layers)

def realignment_training_loop(
    tokenizer,
    model,
    task_dataset: DataLoader,
    realignment_dataset: DataLoader,
    strategy="during",
    evaluation_datasets=None,
    same_language_evaluation_dataset=None,
    evaluation_prefixes=None,
    task_batch_size=4,
    nb_realignment_steps_before=None,
    realignment_batch_size=2,
    learning_rate=2e-5,
    n_epochs=10,
    accumulation_steps=1,
    logging_steps=100,
    log_in_wandb=False,
    result_store=None,
    metric_fn=None,
    realignment_coef=0.1,
    realignment_coef_scheduler=None,
    data_collator=None,
    seed=None,
    epoch_callbacks=None,
    realignment_step_callbacks=None,
    hash_args=None,
    cache_dir=None,
    return_model_hash=False,
    final_prefix="final",
    pretrained_model_fn=None,
    realignment_steps_by_finetuning=1,
    label_key="labels",
    model_name=None,
    checkpoint_path=None,
    task_name=None,
    noaligner=False,
):
    """
    Performs a training loop, with or without realignment

    Arguments:

    - tokenizer
    - model
    - task_dataset: training dataset for the fine-tuning task (must have a length)
    - realignment_dataset: iterable dataset for the realignment auxiliary task
    - strategy: default "during", the realignment strategy (either "baseline" for no realignment, or "after", "before" or "during")
    - evaluation_datasets: optional list of evaluation datasets
    - same_language_evaluation_dataset: optional evaluation dataset on same language as training
    - evaluation_prefixes: optional list of prefixes for evaluation datasets metrics
    - task_batch_size: batch size for the training task (not considering accumulation steps)
    - nb_realignment_steps_before: if set, number of realignment batches to see before fine-tuning, otherwise it is n_epochs times the number of fine-tuning batch.
        Only taken into account if strategy is "before", "before+during" or "after"
    - realignment_batch_size: batch size of the realignment step
    - learning_rate: learning rate for both fine-tuning and realignment
    - n_epochs: number of epochs for the fine-tuning task
    - accumulation_steps: number of accumulation steps for the fine-tuning task
    - logging_steps: int, default 100, number of steps (in term of optimization steps, hence nb of batch / accumulation steps) between each log of training stats
    - log_in_wandb: (deprecated) whether to log training stats in wandb (conditional import), better to use result_store with loggers.WandbResultStore
    - result_store: an instance of loggers.DefaultResultStore that captures the different results obtained along the training (by default it logs them to the console, but it can store them in
        a dictionary for later retrieval)
    - metric_fn: function that gets the metric from the overall predictions and labels
    - realignment_coef: float, default 0.1, the coefficient to apply to the realignment loss
    - realignment_coef_scheduler: a function that takes an integer (the epoch) and return a float, the coefficient to apply to the realignment loss at
        given epochs, overrides realignment_coef
    - data_collator: default None, if None, will default to DataCollatorForTokenClassification(tokenizer)
    - seed
    - epoch_callbacks: (deprecated) optional list of function that takes the model as input and will be called before the first fine-tuning epoch and after each one
    - realignment_step_callbacks: (deprecated) like epoch_callbacks but for each realignment step
    - hash_args: (deprecated) default None, optional string to add to hashing realigned models (only with before strategy), will cache only if it is provided (ideally with model name, id for realignment dataset and commit hash)
        and if cache_dir is provided
    - cache_dir: default None, optional directory for caching models
    - return_model_hash: default False, whether to return the model hash for model saved after realignment (not fine-tuning !!!) useful only if hash_args and cache_dir are specified and if strategy == "before"
    - final_prefix: prefix for metrics in the final evaluation
    - pretrained_model_fn: (deprecated) when the model is cached, function to instantiate the pretrained model from cache_path
    - realignment_steps_by_finetuning: number of realignment optimization steps to perform by fine-tuning steps (useful in 'during' strategy)
    - label_key: (deprecated) the key for the labels in the training input
    """

    # Define a function to log the status of the Embedding space and each RobertaLayer (frozen or unfrozen)
    def log_layer_status(model, model_name):
        if "roberta" in model_name:
            # Handle RoBERTa and XLM-RoBERTa models
            if any(param.requires_grad for param in model.roberta.embeddings.parameters()):
                logging.info("Embedding Space: Unfrozen")
            else:
                logging.info("Embedding Space: Frozen")
            
            logging.info("Word Embeddings: {}".format(
                model.roberta.embeddings.word_embeddings.weight.requires_grad)
            )
            logging.info("Position Embeddings: {}".format(
                model.roberta.embeddings.position_embeddings.weight.requires_grad)
            )
            logging.info("Token Type Embeddings: {}".format(
                model.roberta.embeddings.token_type_embeddings.weight.requires_grad)
            )
            logging.info("LayerNorm: {}".format(
                model.roberta.embeddings.LayerNorm.weight.requires_grad)
            )
            logging.info("Dropout: {}".format(
                any(p.requires_grad for p in model.roberta.embeddings.dropout.parameters())
            ))

            # Log the status of encoder layers
            layers = list(model.roberta.encoder.layer)
            for i, layer in enumerate(layers):
                if any(param.requires_grad for param in layer.parameters()):
                    logging.info(f"RobertaLayer {i}: Unfrozen")
                else:
                    logging.info(f"RobertaLayer {i}: Frozen")

        elif "distilbert" in model_name:
            # Handle DistilBERT model
            if any(param.requires_grad for param in model.distilbert.embeddings.parameters()):
                logging.info("Embedding Space: Unfrozen")
            else:
                logging.info("Embedding Space: Frozen")

            logging.info("Word Embeddings: {}".format(
                model.distilbert.embeddings.word_embeddings.weight.requires_grad)
            )
            logging.info("Position Embeddings: {}".format(
                model.distilbert.embeddings.position_embeddings.weight.requires_grad)
            )
            logging.info("LayerNorm: {}".format(
                model.distilbert.embeddings.LayerNorm.weight.requires_grad)
            )
            logging.info("Dropout: {}".format(
                model.distilbert.embeddings.dropout.p)
            )

            # Log the status of transformer layers
            for i, layer in enumerate(model.distilbert.transformer.layer):
                if any(param.requires_grad for param in layer.parameters()):
                    logging.info(f"TransformerBlock {i}: Unfrozen")
                else:
                    logging.info(f"TransformerBlock {i}: Frozen")

        elif model_name.startswith("bert"):
            if any(param.requires_grad for param in model.bert.embeddings.parameters()):
                logging.info("Embedding Space: Unfrozen")
            else:
                logging.info("Embedding Space: Frozen")

            logging.info("Word Embeddings: {}".format(
                model.bert.embeddings.word_embeddings.weight.requires_grad)
            )
            logging.info("Position Embeddings: {}".format(
                model.bert.embeddings.position_embeddings.weight.requires_grad)
            )
            logging.info("LayerNorm: {}".format(
                model.bert.embeddings.LayerNorm.weight.requires_grad)
            )
            logging.info("Dropout: {}".format(
                model.bert.embeddings.dropout.p)
            )

            # Log the status of transformer layers
            for i, layer in enumerate(model.bert.encoder.layer):
                if any(param.requires_grad for param in layer.parameters()):
                    logging.info(f"TransformerBlock {i}: Unfrozen")
                else:
                    logging.info(f"TransformerBlock {i}: Frozen")

        else:
            logging.warning(f"Model type '{model_name}' not recognized. No logging performed.")

    data_collator = data_collator or DataCollatorForTokenClassification(tokenizer)
    epoch_callbacks = epoch_callbacks or []
    realignment_step_callbacks = realignment_step_callbacks or []

    if log_in_wandb:
        import wandb

    # Put model to GPU if available
    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # Fix random seed for Pytorch and numpy
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        g = None
        seed_worker = None

    # Create dataloader for the fine-tuning task
    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print()
    print(f'RUNNING MODEL: {model_name}')
    print(model)
    print()

    # If needed, create dataloader for re-alignment task
    if strategy.find('baseline') == -1:
        # Note: if this line is modified, hashing args for caching must be checked

        print("Not Using Baseline")
        print()

        realignment_dataloader = DataLoader(
            realignment_dataset,
            shuffle=False,
            batch_size=realignment_batch_size,
            collate_fn=RealignmentAndOtherCollator(
                tokenizer,
                data_collator,
                noaligner=noaligner,
            ),
        )
    else:
        print("Using Baseline")
        realignment_dataloader = None

    training_state = TrainingState.compute_expected_samples(
        strategy,
        task_dataset,
        task_dataloader,
        n_epochs,
        task_batch_size,
        realignment_batch_size,
        accumulation_steps=accumulation_steps,
        nb_realignment_steps_before=nb_realignment_steps_before,
    )

    # If available, create dataloader for evaluation on training language
    if same_language_evaluation_dataset is not None:
        same_language_evaluation_dataloader = DataLoader(
            same_language_evaluation_dataset,
            shuffle=False,
            batch_size=task_batch_size,
            collate_fn=data_collator,
        )

    use_caching = False
    selected_layers = None

    realignment_ckpt = None

    # If strategy is "before" or "before+during", perform realignment before fine-tuning
    if strategy in ["before", "before+during", 
                    "freeze_realign_unfreeze",
                    "freeze_realign_unfreeze_last_half",
                    "freeze_realign_unfreeze_last_6",
                    "freeze_attn",
                    "freeze_ffn",
                   ] or re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy) or re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy) \
                       or re.match(r"before_random_realign_[0-9]+", strategy) \
                           or re.match(r"before_gradual_(topdown|bottomup|random)_[0-9]+", strategy) \
                               or re.match(r"before_oneatatime_(topdown|bottomup|random)_[0-9]+", strategy) \
                                    or re.match(r"high_anisotropy_.+", strategy) \
                                        or re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy) \
                                            or re.search(r"realign_specific_[0-9]+", strategy) \
                                                or "freeze_ffn" in strategy:

        use_caching = cache_dir is not None and hash_args is not None and seed is not None

        learning_rate = learning_rate

        realignment_steps_before = (
            math.ceil(len(task_dataloader) / accumulation_steps) * n_epochs
            if nb_realignment_steps_before is None
            else nb_realignment_steps_before
        )

        if use_caching:
            string_to_hash = (
                hash_args
                + "__"
                + "__".join(
                    [
                        str(learning_rate),
                        str(seed),
                        str(realignment_batch_size),
                        str(realignment_steps_before),
                    ]
                )
            )
            model_hash = hashlib.md5(string_to_hash.encode()).hexdigest()

            cache_path = os.path.join(cache_dir, model_hash)
            training_state_path = os.path.join(cache_dir, f"{model_hash}.json")
            info_path = os.path.join(cache_dir, f"{model_hash}.info")
        else:
            cache_path = None
            training_state_path = None
            info_path = None

        # Note: if this line is modified, hashing args for caching must be checked
        before_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        found_caching = False

        if cache_path is not None and os.path.isfile(training_state_path):
            try:
                with open(training_state_path, "r") as f:
                    other_training_state = TrainingState(**json.load(f))
                    training_state.update_from_other_finetuning(other_training_state)
                found_caching = True
            except json.decoder.JSONDecodeError:
                logging.error(f"Could not decode cached training state. Will not use the cache.")

        if found_caching:
            logging.info(f"Loading cached model: {model_hash}")
            model = (
                pretrained_model_fn(cache_path, ignore_mismatched_sizes=True)
                if pretrained_model_fn is not None
                else model.__class__.from_pretrained(cache_path, ignore_mismatched_sizes=True)
            )
        else:

            # print()
            # print('Realignment Dataloader')
            # for i, batch in enumerate(realignment_dataloader):
            #     # print(batch.keys())
            #     if i == 1:  # This will print the first batch
            #         break

            # print()
            # print('Fine-tuning Dataloader')
            # for i, batch in enumerate(task_dataloader):
            #     # print(batch)
            #     if i == 1:  # This will print the first batch
            #         break
            # print()

            print('')
            print('STARTING REALIGNMENT')
            print()

            if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
                print('Freezing first 6 encoders...')
                for i in range(6):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
                freeze_realign_unfreeze_layers = 6
                print(f'Freezing last {freeze_realign_unfreeze_layers} encoders...')
                
                total_layers = len(model.roberta.encoder.layer)
                for i in range(total_layers - freeze_realign_unfreeze_layers, total_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Freezing first {layers_to_freeze} transformer blocks...')
                for i in range(layers_to_freeze):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Freezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')
            
            if strategy == "freeze_realign_unfreeze" and model_name.startswith("bert"):
                num_layers = len(model.bert.encoder.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Freezing first {layers_to_freeze} encoder blocks...')
                for i in range(layers_to_freeze):
                    for param in model.bert.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
                num_layers = len(model.bert.encoder.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Freezing last {layers_to_freeze} encoder blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.bert.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')
            
            if strategy == "freeze_realign_unfreeze_last_half" and "roberta" in model_name:
                num_layers = len(model.roberta.encoder.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Freezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = False

                print('Freezing done...')

            if strategy == "freeze_attn":
                if "roberta" in model_name:
                    attention_structure = [layer.attention for layer in model.roberta.encoder.layer]
                elif "distilbert" in model_name:
                    attention_structure = [layer.attention for layer in model.distilbert.transformer.layer]
                elif model_name.startswith("bert"):
                    attention_structure = [layer.attention for layer in model.bert.encoder.layer]
                else:
                    raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")

                print(f'Freezing attention structure in transformer blocks...')
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters BEFORE freezing: {trainable_params}")
                for layer_attn in attention_structure:
                    for param in layer_attn.parameters():
                        param.requires_grad = False
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters AFTER freezing: {trainable_params}")
                print('Freezing done...')

            if "freeze_ffn" in strategy:
                if "roberta" in model_name:
                    ffn_structure = [
                        dense for layer in model.roberta.encoder.layer
                        for dense in (layer.intermediate.dense, layer.output.dense)
                    ]
                elif "distilbert" in model_name:
                    ffn_structure = [
                        lin for layer in model.distilbert.transformer.layer
                        for lin in (layer.ffn.lin1, layer.ffn.lin2)
                    ]
                elif model_name.startswith("bert"):
                    ffn_structure = [
                        dense for layer in model.bert.encoder.layer
                        for dense in (layer.intermediate.dense, layer.output.dense)
                    ]
                else:
                    raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")

                print(f'Freezing FFN structure in transformer blocks...')
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters BEFORE freezing: {trainable_params}")
                for layer_ffn in ffn_structure:
                    for param in layer_ffn.parameters():
                        param.requires_grad = False
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters AFTER freezing: {trainable_params}")
                print('Freezing done...')

            if re.match(r"high_anisotropy_.+", strategy) and not re.search(r"(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy):
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
                  
                method = strategy.split("_")[-1]
                layer_to_realign = get_high_ani_layers_for_realignment(model_name, method)
                logging.info(f"High anisotropy layers to be realign: {str(layer_to_realign)}")
                for i, layer in enumerate(layers):
                    if i not in layer_to_realign:
                        for param in layer.parameters():
                            param.requires_grad = False

            if re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy):
                *_, first_layer, last_layer = strategy.split("_")
                first_layer = int(first_layer)
                last_layer = int(last_layer)
                if first_layer == 0:
                    if "roberta" in model_name:
                        embeddings = model.roberta.embeddings
                    elif "distilbert" in model_name:
                        embeddings = model.distilbert.embeddings
                    elif model_name.startswith("bert"):
                        embeddings = model.bert.embeddings
                    else:
                        raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                    for param in embeddings.parameters():
                        param.requires_grad = False
                    first_layer = 1
                for i in range(first_layer - 1, last_layer - 1):
                    if "roberta" in model_name:
                        layers = model.roberta.encoder.layer
                    elif "distilbert" in model_name:
                        layers = model.distilbert.transformer.layer
                    elif model_name.startswith("bert"):
                        layers = model.bert.encoder.layer
                    else:
                        raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                    for param in layers[i].parameters():
                        param.requires_grad = False

            if re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy):
                *_, first_layer, last_layer = strategy.split("_")
                first_layer = int(first_layer)
                last_layer = int(last_layer)
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type /before_realign_only_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                for i, layer in enumerate(layers):
                    if i < first_layer or i >= last_layer:
                        for param in layer.parameters():
                            param.requires_grad = False
                            
            # Realign but choose K layers randomly
            k_layers = None
            if re.match(r"before_random_realign_[0-9]+", strategy):
                *_, k_layers = strategy.split("_")
                k_layers = int(k_layers)
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type /before_realign_only_random_[0-9]+/ is not implemented for model {model_name}")
                selected_layers = select_layers_for_realignment(len(layers), k_layers)
                logging.info(f"Current selected layers! {str(selected_layers)}")
                for i, layer in enumerate(layers):
                    if i not in selected_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
            
            if re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy):
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
                
                #Extract configg
                match = re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy)
                num_realign_lays, noembs_flag, position_type = match.groups()

                selected_indices = []
                start_idx = 0
                if noembs_flag:
                    if noembs_flag == "_noembs":
                        layers = layers[1:] # Skip the embs layer
                    elif noembs_flag == "withembs":
                        selected_indices = [0] # Add the embs layer
                        start_idx = 1

                #Get number of layers to freeze
                if num_realign_lays == "half":
                    num_realign_lays = len(layers) // 2 
                elif num_realign_lays == "twothird":
                    num_realign_lays = (2 * len(layers)) // 3    
                elif num_realign_lays == "onethird":
                    num_realign_lays = len(layers) // 3
                elif num_realign_lays == "fivesixth":
                    num_realign_lays = (5 * len(layers)) // 6    
                elif num_realign_lays == "onesixth":
                    num_realign_lays = len(layers) // 6

                if noembs_flag == "withembs":
                    num_realign_lays -= 1

                total_layers = len(layers)
                if position_type:
                    if position_type == "_adjacent":
                        #Select adjacent layers
                        start = random.randint(start_idx, total_layers - num_realign_lays)
                        selected_indices.extend(list(range(start, start + num_realign_lays)))
                    elif position_type == "_discrete":
                        #Select discrete layers
                        possible_indices = list(range(total_layers))
                        for i in range(num_realign_lays):
                            index = random.choice(possible_indices)
                            selected_indices.append(index)
                            possible_indices = [idx for idx in possible_indices if idx not in {index, index + 1, index - 1}]
                        selected_indices = sorted(selected_indices)
                else:
                    selected_indices.extend(sorted(random.sample(range(total_layers), num_realign_lays)))
                logging.info(f"Layers to be realign: {str(selected_indices)}")
                
                #Freezing 
                for i, layer in enumerate(layers):
                    if i not in selected_indices:
                        for param in layer.parameters():
                            param.requires_grad = False

            if re.search(r"realign_specific_[0-9]+", strategy):
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
                
                #Extract indices
                match = re.search(r"realign_specific_[0-9]+", strategy)
                _, selected_indices = match.group(0).split("_")
                selected_indices = sorted([int(i) for i in selected_indices])
                logging.info(f"Layers to be realign: {str(selected_indices)}")
                
                #Freezing 
                for i, layer in enumerate(layers):
                    if i not in selected_indices:
                        for param in layer.parameters():
                            param.requires_grad = False

            log_layer_status(model, model_name)

            # Check for existing realignment ckpt
            
            realignment_checkpoint_prefix_name = None
            if checkpoint_path:
                realignment_checkpoint_prefix_name = f"realignment_{model_name}_seed_{seed}"
                iter_num = str(realignment_steps_before - 1) if realignment_steps_before else "*"
                pattern = os.path.join(checkpoint_path, f"{realignment_checkpoint_prefix_name}_iter_{iter_num}.ckpt")
                matching_files = glob.glob(pattern)
                realignment_ckpt = None
                if matching_files:
                    realignment_ckpt_file = sorted(matching_files)[-1]
                    realignment_ckpt = torch.load(realignment_ckpt_file, map_location=torch.device('cpu'))
                    partial_state_dict = {k: v for k, v in realignment_ckpt['model_state_dict'].items() if not k.startswith("classifier")}
                    try:
                        missing_keys, unexpected_keys = model.load_state_dict(partial_state_dict, strict=False)
                        logging.info(f"Found saved realignment checkpoint. Realignment checkpoint loaded {realignment_ckpt_file}")
                        logging.warning(f"Missing keys: {missing_keys}")
                        logging.warning(f"Unexpected keys: {unexpected_keys}")
                    except Exception as e:
                        logging.warning(f"Unable to load realignment ckpt. Error: {e}")
                        del realignment_ckpt, realignment_ckpt_file, partial_state_dict
                        realignment_ckpt = None
                    # This frees up unused memory
                    del realignment_ckpt["model_state_dict"], partial_state_dict
            if not realignment_ckpt:
                training_state = epoch_loop(
                    model,
                    before_optimizer,
                    task_dataloader=None,
                    realignment_dataloader=realignment_dataloader,
                    task_accumulation_steps=1,
                    logging_steps=logging_steps,
                    log_in_wandb=log_in_wandb,
                    result_store=result_store,
                    nb_iter=realignment_steps_before,
                    realignment_step_callbacks=realignment_step_callbacks,
                    training_state=training_state,
                    log_first_sample=True,
                    realignment_steps_by_finetuning=realignment_steps_by_finetuning,
                    model_name=model_name,
                    strategy=strategy,
                    checkpoint_path=checkpoint_path,
                    checkpoint_prefix_name=realignment_checkpoint_prefix_name,
                )

                res = training_state.log_state()
                if log_in_wandb:
                    wandb.log(res)
                if result_store:
                    result_store.log(res)

            if cache_path is not None:
                logging.info(f"Saving realigned model: {model_hash}")
                model.save_pretrained(cache_path)

                with open(os.path.join(cache_path, "info.txt"), "w") as f:
                    f.write(string_to_hash + "\n")

                with open(training_state_path, "w") as f:
                    json.dump(dataclasses.asdict(training_state), f)

                with open(info_path, "w") as f:
                    f.write(hash_args + "\n")

            print('')
            print('DONE REALIGNMENT')
            print()

            if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
                print('Unfreezing first 6 encoders...')
                for i in range(6):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            
            if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
                print('Unfreezing last 6 encoders...')
                
                total_layers = len(model.roberta.encoder.layer)
                for i in range(total_layers - 6, total_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Unfreezing first {layers_to_freeze} transformer blocks...')
                for i in range(layers_to_freeze):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
                num_layers = len(model.distilbert.transformer.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Unfreezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')
            
            if strategy == "freeze_realign_unfreeze" and model_name.startswith("bert"):
                num_layers = len(model.bert.encoder.layer)
                layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

                print(f'Unfreezing first {layers_to_freeze} encoder blocks...')
                for i in range(layers_to_freeze):
                    for param in model.bert.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
                num_layers = len(model.bert.encoder.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Unfreezing last {layers_to_freeze} encoder blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.bert.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_realign_unfreeze_last_half" and "roberta" in model_name:
                num_layers = len(model.roberta.encoder.layer)
                layers_to_freeze = num_layers // 2  # Number of layers to freeze

                # Calculate the starting index for freezing (freezing the last half of the layers)
                start_freezing_from_layer = num_layers - layers_to_freeze

                print(f'Unfreezing last {layers_to_freeze} transformer blocks...')
                for i in range(start_freezing_from_layer, num_layers):
                    for param in model.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True

                print('Unfreezing done...')

            if strategy == "freeze_attn":
                print(f'Unfreezing attention structure in transformer blocks...')
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters BEFORE unfreezing: {trainable_params}")
                for layer_attn in attention_structure:
                    for param in layer_attn.parameters():
                        param.requires_grad = True
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters AFTER unfreezing: {trainable_params}")
                print('Unfreezing done...')

            if "freeze_ffn" in strategy:
                print(f'Unfreezing FFN structure in transformer blocks...')
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters BEFORE unfreezing: {trainable_params}")
                for layer_ffn in ffn_structure:
                    for param in layer_ffn.parameters():
                        param.requires_grad = True
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"___Total trainable parameters AFTER unfreezing: {trainable_params}")
                print('Unfreezing done...')

            if re.match(r"high_anisotropy_.+", strategy) and not re.search(r"(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy):
                print(f"{str(layer_to_realign)} have been realigned. Unfreezing others...")
                for i, layer in enumerate(layers):
                    if i not in layer_to_realign:
                        for param in layer.parameters():
                            param.requires_grad = True
                print('Unfreezing done...')
            
            if re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy):
                *_, first_layer, last_layer = strategy.split("_")
                first_layer = int(first_layer)
                last_layer = int(last_layer)
                if first_layer == 0:
                    if "roberta" in model_name:
                        embeddings = model.roberta.embeddings
                    elif "distilbert" in model_name:
                        embeddings = model.distilbert.embeddings
                    elif model_name.startswith("bert"):
                        embeddings = model.bert.embeddings
                    else:
                        raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                    for param in embeddings.parameters():
                        param.requires_grad = True
                    first_layer = 1
                for i in range(first_layer - 1, last_layer - 1):
                    if "roberta" in model_name:
                        layers = model.roberta.encoder.layer
                    elif "distilbert" in model_name:
                        layers = model.distilbert.transformer.layer
                    elif model_name.startswith("bert"):
                        embeddings = model.bert.encoder.layer
                    else:
                        raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                    for param in layers[i].parameters():
                        param.requires_grad = True
            
            if re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy):
                # TODO: Create new strategy
                *_, first_layer, last_layer = strategy.split("_")
                first_layer = int(first_layer)
                last_layer = int(last_layer)
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type /before_realign_only_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
                for i, layer in enumerate(layers):
                    if i < first_layer or i >= last_layer:
                        for param in layer.parameters():
                            param.requires_grad = True
                            
            # Realign but choose K layers randomly
            if re.match(r"before_random_realign_[0-9]+", strategy):
                if "roberta" in model_name:
                    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
                elif "distilbert" in model_name:
                    layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
                elif model_name.startswith("bert"):
                    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
                else:
                    raise NotImplementedError(f"Strategy of type /before_realign_only_random_[0-9]+/ is not implemented for model {model_name}")
                for i, layer in enumerate(layers):
                    for param in layer.parameters():
                        param.requires_grad = True

            if re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy) or re.search(r"realign_specific_[0-9]+", strategy):                
                logging.info(f"Unfreezing other layers than {str(selected_indices)}")
                for i, layer in enumerate(layers):
                    if i not in selected_indices:
                        for param in layer.parameters():
                            param.requires_grad = True
                print('Unfreezing done...')

    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * len(task_dataloader) * n_epochs),
        num_training_steps=len(task_dataloader) * n_epochs,
    )

    for callback in epoch_callbacks:
        callback(model)

    print()
    print('STARTING FINETUNING')
    print()

    if strategy in ["during_freeze_realign_unfreeze", 
                    "baseline_freeze_realign_unfreeze"] and "roberta" in model_name:
        print('Freezing first 6 encoders...')
        for i in range(6):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy in ["baseline_freeze_realign_unfreeze_last_6",
                    "during_freeze_realign_unfreeze_last_6"] and "roberta" in model_name:
        total_layers = len(model.roberta.encoder.layer)
        for i in range(total_layers - 6, total_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    if strategy == "during_freeze_realign_unfreeze" and "distilbert" in model_name:
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} transformer blocks...')
        for i in range(layers_to_freeze):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze" and model_name.startswith("bert"):
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} encoder blocks...')
        for i in range(layers_to_freeze):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} encoder blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    
    if strategy == "during_freeze_realign_unfreeze_last_half" and "roberta" in model_name:
        num_layers = len(model.roberta.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    

    realignment_optimizer = None
    realignment_scheduler = None
    realignment_ignore_parameters = []
    if strategy.startswith("during_partial_freeze"):


        if "roberta" in model_name:
            n_layers = len(model.roberta.encoder.layer)
            encoder_prefix = "roberta.encoder.layer"
            embedding_prefix = "roberta.embeddings"
        elif "distilbert" in model_name:
            n_layers = len(model.distilbert.transformer.layer)
            encoder_prefix = "distilbert.transformer.layer"
            embedding_prefix = "distilbert.embeddings"
        elif model_name.startswith("bert"):
            n_layers = len(model.bert.encoder.layer)
            encoder_prefix = "bert.encoder.layer"
            embedding_prefix = "bert.embeddings"
        else:
            raise NotImplementedError(f"during_partial_freeze_* strategies are not implemented for model {model}")
        
        if strategy.endswith("front"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2)]
        elif strategy.endswith("back"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2, n_layers)]
        elif strategy.endswith("none"):
            prefixes_to_ignore = []
        elif re.match(r"during_partial_freeze_[0-9]+_[0-9]+", strategy):
            *_, first_layer, last_layer = strategy.split("_")
            first_layer = int(first_layer)
            last_layer = int(last_layer)
            prefixes_to_ignore = []
            if first_layer == 0:
                prefixes_to_ignore.append(embedding_prefix)
                first_layer = 1
            for i in range(first_layer - 1, last_layer - 1):
                prefixes_to_ignore.append(f"{encoder_prefix}.{i}")
        else:
            raise NotImplementedError(f"Unrecognized strategy {strategy}")
        
        for name, param in model.named_parameters():
            if any(map(lambda x: name.startswith(x), prefixes_to_ignore)):
                realignment_ignore_parameters.append(name)
        
        if not strategy.endswith("none"):
            assert len(realignment_ignore_parameters) > 0

    log_layer_status(model, model_name)
    
    if not isinstance(evaluation_datasets, list) and evaluation_datasets is not None:
        evaluation_datasets = [evaluation_datasets]

    finetuning_ckpt = None
    checkpoint_epoch = None
    if checkpoint_path:
        finetuning_checkpoint_prefix_name = f"finetuning_{model_name}_{task_name}_seed_{seed}"
        pattern = os.path.join(checkpoint_path, f"{finetuning_checkpoint_prefix_name}_epoch_*_iter_*.ckpt")
        matching_files = glob.glob(pattern)
        if matching_files:
            finetuning_ckpt_file = sorted(matching_files)[-1]
            checkpoint_epoch = int(re.search(r"epoch_([0-9]+)", finetuning_ckpt_file).group(1))
            finetuning_ckpt = torch.load(finetuning_ckpt_file, map_location=torch.device('cpu'))
            try:
                missing_keys, unexpected_keys = model.load_state_dict(finetuning_ckpt['model_state_dict'], strict=True)
                logging.info(f"Found saved finetuning checkpoint. finetuning checkpoint loaded {finetuning_ckpt_file}")
                logging.warning(f"Missing keys: {missing_keys}")
                logging.warning(f"Unexpected keys: {unexpected_keys}")
            except Exception as e:
                logging.warning(f"Unable to load finetuning ckpt. Error: {e}")
                del finetuning_ckpt, finetuning_ckpt_file
                finetuning_ckpt = None
            # This frees up unused memory
            del finetuning_ckpt["model_state_dict"]

    for i in range(n_epochs):
        if checkpoint_epoch is not None and i <= checkpoint_epoch:
            logging.info(f"Skipping epoch {i} as it is already completed in checkpoint.")
            continue

        if i == n_epochs - 1:
            logging.info(f"Saving checkpoint for epoch {i}")
            ft_checkpoint_path = checkpoint_path
            ft_prefix_name = f"finetuning_{model_name}_{task_name}_seed_{seed}_epoch_{i}"
        else:
            ft_checkpoint_path = ft_prefix_name = None
        
        training_state = epoch_loop(
            model,
            optimizer,
            scheduler=scheduler,
            task_dataloader=task_dataloader,
            realignment_dataloader=realignment_dataloader
            if "during" in strategy or strategy == "staged"
            else None,
            realignment_optimizer=realignment_optimizer,
            realignment_scheduler=realignment_scheduler,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            realignment_coef=realignment_coef
            if realignment_coef_scheduler is None
            else realignment_coef_scheduler(i),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            log_first_sample=i == 0,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
            separate_backward=strategy == "during_separate_backward" or bool(realignment_ignore_parameters),
            realignment_ignore_parameters=realignment_ignore_parameters,
            checkpoint_path=ft_checkpoint_path,
            checkpoint_prefix_name=ft_prefix_name,
        )
        for callback in epoch_callbacks:
            callback(model)

        res = training_state.log_state()
        if realignment_ckpt:
            res["realignment_steps"] = realignment_ckpt['nb_iter']
            res["realignment_loss"] = realignment_ckpt['loss']
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

        if evaluation_datasets is not None:
            res = evaluate_several_token_classification(
                tokenizer,
                model,
                evaluation_datasets,
                batch_size=task_batch_size,
                prefixes=evaluation_prefixes,
                overall_prefix="eval",
                metric_fn=metric_fn,
                collator=data_collator,
                label_key=label_key,
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)
        if same_language_evaluation_dataset is not None:
            res = evaluate_token_classification(
                model, same_language_evaluation_dataloader, prefix="eval_same", metric_fn=metric_fn
            )
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)

    print()
    print('DONE FINETUNING')
    print()

    if strategy == "after":
        after_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        training_state = epoch_loop(
            model,
            after_optimizer,
            task_dataloader=None,
            realignment_dataloader=realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            nb_iter=(
                len(task_dataloader) * n_epochs
                if nb_realignment_steps_before is None
                else nb_realignment_steps_before * accumulation_steps
            ),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
        )
        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
        for callback in epoch_callbacks:
            callback(model)

    if evaluation_datasets is not None:
        res = evaluate_several_token_classification(
            tokenizer,
            model,
            evaluation_datasets,
            batch_size=task_batch_size,
            prefixes=evaluation_prefixes,
            overall_prefix=f"{final_prefix}_eval",
            metric_fn=metric_fn,
            collator=data_collator,
            label_key=label_key,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
    if same_language_evaluation_dataset is not None:
        res = evaluate_token_classification(
            model,
            same_language_evaluation_dataloader,
            prefix=f"{final_prefix}_eval_same",
            metric_fn=metric_fn,
        )
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

    if return_model_hash and use_caching:
        return model_hash
