import os
import json
import itertools
import torch
import logging
import math
from typing import Optional
import random
import numpy as np
from tqdm import tqdm
from typing import Literal
from transformers.optimization import get_scheduler

from multilingual_eval.datasets.weighted_data_utils import weighted_multilingual_batch
from multilingual_eval.training.utils import bring_batch_to_model, get_next_or_restart
from multilingual_eval.training.states import TrainingState

from multilingual_eval.models.simplified import SimplifiedModelForRealignment


def extract_adapter_state_dict(model, use_adapter=False, adapter_approach="same"):
    """
    Extracts the adapter state dict from a model.
    
    If the model is using adapters (use_adapter=True), this tries to extract only the adapter-related parameters
    (and realignment_head if present).
    
    Args:
        model: A model that may be wrapped with PEFT adapters, or SimplifiedModelForRealignment wrapping one.
        use_adapter (bool): Whether adapters are being used.
        adapter_approach (str): The strategy used ("same", "separate", "realign_only").
        
    Returns:
        dict: State dict containing adapter weights only if model has adapters, else full state dict.
    """
    if not use_adapter:
        return model.state_dict()

    try:
        from peft import PeftModel
        
        # Unwrap SimplifiedModelForRealignment for check
        model_to_check = model.model if isinstance(model, SimplifiedModelForRealignment) else model
        is_peft = isinstance(model_to_check, PeftModel)
        
        # Determine if we should save full state
        # For 'realign_only', if we are in finetuning phase (not PeftModel anymore), we must save full state.
        if adapter_approach == "realign_only" and not is_peft:
            return model.state_dict()
            
        full_state_dict = model.state_dict()
        adapter_state_dict = {}
        
        # Filter for adapter keys and realignment head
        for key, value in full_state_dict.items():
            if any(x in key for x in ["lora_"]):
                adapter_state_dict[key] = value
                
        return adapter_state_dict if adapter_state_dict else full_state_dict

    except ImportError:
        # PEFT not installed, return full state dict
        return model.state_dict()


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

def epoch_loop(
    model,
    optimizer,
    scheduler=None,
    task_dataloader=None,
    realignment_dataloader=None,
    realignment_optimizer=None,
    realignment_scheduler=None,
    task_accumulation_steps=1,
    realignment_steps_by_finetuning=1,
    logging_steps=100,
    log_in_wandb=False,
    result_store=None,
    nb_iter=None,
    start_iter=0,
    extra_realignment_steps_checkpoints=None,
    realignment_coef=1.0,
    realignment_step_callbacks=None,
    training_state: Optional[TrainingState] = None,
    log_first_sample=False,
    parallelism=False,
    separate_backward=False,
    realignment_ignore_parameters: Optional[list] = None,
    model_name=None,
    strategy=None,
    checkpoint_path=None,
    checkpoint_prefix_name="model",
    use_adapter=False,
    adapter_approach="same",
    adapter_config=None,
):
    """
    Function to perform an epoch of training, with specific task samples and/or realignment task samples

    Arguments:

    - optimizer
    - task_dataloader: the dataloader for the training task (if None, only realignment is performed), default is None
    - realignment_dataloader: the dataloader for the realignment task (if None, only main task is trained for), default is None
    - task_accumulation_steps: int, accumulation steps for the main task
    - logging_steps: int, default 100, number of steps (in term of optimization steps, hence nb of batch / accumulation steps) between each log of training stats
    - log_in_wandb: whether to log training stats in wandb (conditional import)
    - nb_iter: optional int, default to None, number of iteration to perform if task_dataloader is not provided
    - realignment_coef: float, default 1., coefficient to apply to the realignment loss
    """
    def save_checkpoint():
        import peft


        # Checkpoint
        if checkpoint_path:
            logging.info(f"Saved model at {i}/{nb_iter} at {checkpoint_prefix_name}_iter_{i}.ckpt")
            model_state_dict = extract_adapter_state_dict(model, use_adapter, adapter_approach)
            torch.save({
                'iter': i,
                'nb_iter': nb_iter,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu()),
                'use_adapter': use_adapter,
                'adapter_approach': adapter_approach,
                'adapter_config': adapter_config,
            }, os.path.join(checkpoint_path, f"{checkpoint_prefix_name}_iter_{i}.ckpt"))
    
    class ContinueNextBatch(Exception):
        pass

    realignment_step_callbacks = realignment_step_callbacks or []
    if realignment_dataloader is None and task_dataloader is None:
        raise Exception(
            "Both task_dataloader and realignment_dataloader cannot be None, we need to train on at least one dataloader"
        )

    if task_dataloader is None and nb_iter is None:
        raise Exception(
            f"If task_dataloader is not provided (got {task_dataloader}), you should provide nb_iter (got {nb_iter})"
        )

    if nb_iter is not None and task_dataloader is not None:
        logging.warning(
            f"nb_iter was provided ({nb_iter}) but so was task_dataloader. nb_iter will be ignored."
        )
    if not separate_backward and bool(realignment_ignore_parameters):
        raise Exception(
            f"If realignment_ignore_parameters ({bool(realignment_ignore_parameters is not None)}) is set "
            + "then separate_backward must be set to True (was set to False)"
        )

    if task_dataloader is not None:
        nb_iter = len(task_dataloader)
        
    # --- Helper Functions ---
    # --- Preprocess Freeze Schedule ---
    # Convert schedule to a list of actions sorted by step/epoch
    import re
    
    layers = None
    scheduling_freeze = False
    phase = None
    direction = None
    progress = nb_iter

    if strategy and (re.match(r"before_(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy) or re.search(r"(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy)):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if model_name:
            if "roberta" in model_name:
                layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
            elif "distilbert" in model_name:
                layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
            elif model_name.startswith("bert"):
                layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise ValueError("Provide model_name please for this case")
        
        if strategy.startswith("high_anisotropy"):
            pattern = r"high_anisotropy_(\w+)_((gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+)"
            match = re.search(pattern, strategy)
            method = match.group(1)
            strategy = "_".join(["dummy", match.group(2)])
            logging.info(f"Method: {method}")
            logging.info(f"Strategy: {strategy}")
            layer_to_realign = get_high_ani_layers_for_realignment(model_name, method)
            layers = [layers[i] for i in layer_to_realign]
        
        # Extract the direction (topdown/bottomup) and progress value
        phase = strategy.split("_")[1]
        direction = strategy.split("_")[2]
        progress = int(strategy.split("_")[3])

        num_layers = len(layers)
        num_unfrozen_layers = max(1, num_layers // progress)  # Ensure at least 1 layer is unfrozen
        
         # Apply freezing/unfreezing based on direction
        if direction == "topdown":
            # Top-down: Unfreeze the last `num_unfrozen_layers` layers
            for i, layer in enumerate(layers):
                if i >= num_layers - num_unfrozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        elif direction == "bottomup":
            # Bottom-up: Unfreeze the first `num_unfrozen_layers` layers
            for i, layer in enumerate(layers):
                if i < num_unfrozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        elif direction == "random":
            selected_layers = select_layers_for_realignment(len(layers), progress)
            logging.info(f"Current selected layers! {str(selected_layers)}")
            for i, layer in enumerate(layers):
                if i in selected_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        scheduling_freeze = True
        
        for i, layer in enumerate(layers):
            if any(param.requires_grad for param in layer.parameters()):
                logging.info(f"RobertaLayer {i}: Unfrozen")
            else:
                logging.info(f"RobertaLayer {i}: Frozen")

    model.train()
    if log_in_wandb:
        import wandb

    if realignment_dataloader is not None:
        realignment_iterator = iter(realignment_dataloader)

    nb_batch = math.ceil(nb_iter / task_accumulation_steps)

    progress_bar = tqdm(total=nb_batch)

    optimizer.zero_grad()
    if realignment_optimizer:
        realignment_optimizer.zero_grad()

    for i, batch in (
        enumerate(task_dataloader)
        if task_dataloader is not None
        else enumerate(itertools.repeat(None, nb_iter))
    ):
        try:
            if scheduling_freeze and phase and direction and i > 0 and i % (nb_iter // progress) == 0:
                save_checkpoint()
                
                # Calculate the next set of layers to unfreeze
                freeze_step = i // (nb_iter // progress)
                num_layers = len(layers)
                num_unfrozen_layers = max(1, ((freeze_step + 1) * num_layers) // progress)  # Ensure at least 1 layer is unfrozen
                if direction == "topdown":
                    # Top-down: Unfreeze the last `num_unfrozen_layers` layers
                    for i, layer in enumerate(layers):
                        if phase == "gradual":
                            if i >= num_layers - num_unfrozen_layers:
                                for param in layer.parameters():
                                    param.requires_grad = True
                        elif phase == "oneatatime":
                            prev_unfrozen_start = num_layers - ((freeze_step * num_layers) // progress)
                            new_unfrozen_start = num_layers - num_unfrozen_layers
                            if prev_unfrozen_start <= i:
                                for param in layer.parameters():
                                    param.requires_grad = False  # Refreeze previously unfrozen layers
                            elif new_unfrozen_start <= i:
                                for param in layer.parameters():
                                    param.requires_grad = True  # Unfreeze new layers

                elif direction == "bottomup":
                    # Bottom-up: Unfreeze the first `num_unfrozen_layers` layers
                    for i, layer in enumerate(layers):
                        if phase == "gradual":
                            if i < num_unfrozen_layers:
                                for param in layer.parameters():
                                    param.requires_grad = True
                        elif phase == "oneatatime":
                            prev_unfrozen_start = freeze_step * num_layers // progress
                            new_unfrozen_start = num_unfrozen_layers
                            if i < prev_unfrozen_start:
                                for param in layer.parameters():
                                    param.requires_grad = False  # Refreeze previously unfrozen layers
                            elif i < new_unfrozen_start:
                                for param in layer.parameters():
                                    param.requires_grad = True  # Unfreeze new layers
                elif direction == "random":
                    selected_layers = select_layers_for_realignment(len(layers), progress)
                    logging.info(f"Current selected layers! {str(selected_layers)}")
                    for i, layer in enumerate(layers):
                        if i in selected_layers:
                            for param in layer.parameters():
                                param.requires_grad = True
                        else:
                            for param in layer.parameters():
                                param.requires_grad = False
                                
                for i, layer in enumerate(layers):
                    if any(param.requires_grad for param in layer.parameters()):
                        logging.info(f"RobertaLayer {i}: Unfrozen")
                    else:
                        logging.info(f"RobertaLayer {i}: Frozen")
            
            if i % task_accumulation_steps == 0:
                
                accumulated_steps = 0
                total_loss = 0
                task_loss = 0
                realignment_loss = 0

                if realignment_dataloader is not None:
                    for _ in range(realignment_steps_by_finetuning):
                        realignment_iterator, realignment_batch, restarted = get_next_or_restart(
                            realignment_dataloader, realignment_iterator
                        )

                        if training_state is not None:
                            training_state.has_restarted = training_state.has_restarted or restarted

                            if not training_state.has_restarted:
                                training_state.nb_realignment_samples_seen_before_restart += (
                                    realignment_batch["left_input_ids"].shape[0]
                                )

                            training_state.nb_realignment_samples_seen += realignment_batch[
                                "left_input_ids"
                            ].shape[0]
                            training_state.nb_realignment_steps_seen += 1
                        
                        if i <= start_iter: 
                            raise ContinueNextBatch

                        realignment_batch = bring_batch_to_model(realignment_batch, model)
                        outputs = model(**realignment_batch, return_dict=True)

                        realignment_loss += (
                            realignment_coef / realignment_steps_by_finetuning
                        ) * outputs.loss

                    if separate_backward or realignment_optimizer:
                        realignment_loss.backward()

                    if realignment_optimizer:
                        realignment_optimizer.step()
                        realignment_optimizer.zero_grad()

                        if realignment_scheduler:
                            realignment_scheduler.step()

                        optimizer.zero_grad()
                    
                    if realignment_ignore_parameters:
                        for name, param in model.named_parameters():
                            if name in realignment_ignore_parameters:
                                param.grad = None
                    
                    total_loss += realignment_loss

                    if isinstance(extra_realignment_steps_checkpoints, list) and i + 1 in extra_realignment_steps_checkpoints:
                        save_checkpoint()

        except ContinueNextBatch:
            progress_bar.update()
            continue

        if batch is not None:

            if parallelism and torch.cuda.device_count() > 1:
                outputs = torch.nn.parallel.data_parallel(model, None, module_kwargs=batch)
                tmp_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                task_loss += tmp_loss.mean()
            else:
                batch = bring_batch_to_model(batch, model)
                outputs = model(**batch)
                task_loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            accumulated_steps += 1

        if i % task_accumulation_steps == task_accumulation_steps - 1 or i == nb_iter - 1:
            if training_state is not None and batch is not None:
                training_state.nb_finetuning_steps_seen += 1

            task_loss /= max(1, accumulated_steps)
            total_loss += task_loss

            if accumulated_steps > 0 and (separate_backward or realignment_optimizer):
                task_loss.backward()
            else:
                total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            if realignment_dataloader is not None:
                for callback in realignment_step_callbacks:
                    callback(model)

            progress_bar.update()

            if logging_steps is not None and (i // task_accumulation_steps) % logging_steps == 0:
                if training_state is not None:
                    res = training_state.log_state()
                else:
                    batch_seen = math.ceil(i / task_accumulation_steps)

                    logging.info(f"batch: {i}/{nb_batch} loss : {total_loss} {progress_bar}")
                    res = None

                if log_in_wandb:
                    wandb.log(
                        {
                            **(res if res is not None else {"train_step": batch_seen}),
                            "train_loss": total_loss if total_loss == 0  else float(total_loss.detach().cpu()),
                            "realignment_loss": realignment_loss if realignment_loss == 0 else float(realignment_loss.detach().cpu()),
                            "task_loss": task_loss if task_loss == 0 else float(task_loss.detach().cpu()),
                        }
                    )
                if result_store:
                    result_store.log(
                        {
                            **(res if res is not None else {"train_step": batch_seen}),
                            "train_loss": total_loss if total_loss == 0  else float(total_loss.detach().cpu()),
                            "realignment_loss": realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu()),
                            "task_loss": task_loss if task_loss == 0  else float(task_loss.detach().cpu()),
                        }
                    )

    progress_bar.close()
    
    if strategy and (re.match(r"before_gradual_random_[0-9]+", strategy) or re.match(r"before_(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy)):
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = True
    
    # Checkpoint
    print(adapter_config)
    save_checkpoint()

    return training_state


def fine_tuning_loop(
    model,
    dataset,
    data_collator,
    task_name: str,
    batch_size=32,
    accumulation_steps=1,
    seed=None,
    steps=2_000,
    learning_rate=2e-5,
):

    print()
    print('INSIDE FINE TUNING LOOP')

    model.train()
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * steps),
        num_training_steps=steps,
    )
    iterator = iter(dataloader)

    n_epochs = 0

    for i in range(steps):
        loss = 0
        optimizer.zero_grad()

        for j in range(accumulation_steps):
            iterator, batch, restarted = get_next_or_restart(dataloader, iterator, name=task_name)

            if restarted:
                n_epochs += 1

            batch = bring_batch_to_model(batch, model)
            outputs = model(**batch)
            loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss.backward()
        optimizer.step()
        scheduler.step()


def realignment_epoch(
    model, iterator, dataloader, optimizer, scheduler=None, batch_size=16, steps=2_000
):
    model.train()
    for i in range(steps):
        optimizer.zero_grad()

        iterator, batch, restarted = get_next_or_restart(dataloader, iterator, "realignment")

        batch = bring_batch_to_model(batch, model)
        outputs = model(**batch, return_dict=True)
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def weighted_selection_realignment(
    model,
    tokenizer,
    realignment_optimizer,
    alignment_datasets: list,
    noaligner,
    meta_weight=None,
    meta_loss_type="micro", 
    meta_learning_rate=1e-2, 
    inner_batches_before_outer=5, 
    lbsmoothing_eps=0,
    with_regularization=False, 
    lambda_entropy=1e-3, 
    noise_mixing_strat=None,
    softmax_temp=1,
    realignment_steps=1000,
    start_iter=-1,
    extra_realignment_steps_checkpoints=None,
    realignment_coef=1.0,
    realignment_steps_by_finetuning=1,
    batch_size=32,
    logging_steps=100,
    result_store=None,
    training_state: Optional[TrainingState] = None,
    checkpoint_path=None,
    checkpoint_prefix_name=None,
    use_adapter=False,
    adapter_approach="",
    adapter_config=None,
    to_train: Literal["all","only_meta","only_meta_closed","only_weights"] = "all",
    data_collator=None,
):
    '''
    GOAL:
    - meta_weight is used for sampling data
    - meta_loss is calculate through realignment loss (soft loss) 
    - maximize weights for languages that contribute the most to the alignment loss

    meta_weight: Weight of each language in the realignment list
    meta_learning_rate: Learning rate for meta weight (Should be larger than model's lr)
    inner_batches_before_outer: Number of inner batches the model sees before the meta weight loss backward
    realignment_steps: Number of realignment step to perform on the model (inner loop)
    model_learning_rate: Learning rate for realignment process
    lbsmoothing_eps: Smooth out the sampling probs by mixing in a uniform distribution by epsilon percentage.
    with_regularization: Whether to include language weight entropy regularization to the meta loss 
    lambda_entropy: Proportion of entropy attribute to meta loss if regularization is enabled
    noise_mixing_strat: Proportion of weighted sample within a batch
    softmax_temp: Scaling the weights by the temparature
    meta_loss_type: Either micro (treating every example in the batch equally) to macro (treating every language in batches equally)
    '''

    def save_checkpoint():
        realignment_iter = i + j
        ckpt_file_name = f"{checkpoint_prefix_name}_iter_{realignment_iter}.ckpt"
        logging.info(f"Saved model at {realignment_iter}/{realignment_steps} at {ckpt_file_name}")
        model_state_dict = extract_adapter_state_dict(model, use_adapter, adapter_approach)
        torch.save({
            'iter': realignment_iter,
            'nb_iter': realignment_steps,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': realignment_optimizer.state_dict(),
            'loss': realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu()),
            "meta_weight": meta_weight_detach,
            "meta_loss_total": meta_loss_detach,
            'use_adapter': use_adapter,
            'adapter_approach': adapter_approach,
            'adapter_config': adapter_config,
        }, os.path.join(checkpoint_path, ckpt_file_name))

    if not realignment_steps:
        raise Exception("realignment_steps must be provided")
    # meta_weight = meta_weight or torch.nn.Parameter(0.01 * torch.randn(len(alignment_datasets), device="cuda"))
    # Uniform initialization
    meta_weight = meta_weight if meta_weight is not None else torch.nn.Parameter(torch.ones(len(alignment_datasets), device="cuda"))

    # ---- Static closed-form meta-weight computation ----
    # In the static (decoupled) setting there is a closed-form solution for the
    # meta-weights: run a single pass over the data with uniform sampling,
    # average the per-language realignment losses (no backprop), and set
    # meta_weight to those losses so the subsequent
    # softmax(meta_weight / softmax_temp) yields softmax of the per-language
    # losses (the desired sampling distribution). No optimizer is needed.
    if to_train == "only_meta_closed":
        # from datasets import interleave_datasets
        # from torch.utils.data import DataLoader
        # from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset
        # from multilingual_eval.datasets.realignment_dataset import (
        #     RealignmentAndOtherCollator,
        # )
        # from multilingual_eval.datasets.dispatch_datasets import (
        #     collator_fn,
        # )

        num_langs = len(alignment_datasets)
        device = meta_weight.device
        per_lang_loss_sum = torch.zeros(num_langs, device=device)
        per_lang_count = torch.zeros(num_langs, device=device)
        uniform_probs = torch.ones(num_langs, device=device) / num_langs

        ## Convert back 
        # logging.info("Converting alignment datasets list to Torch compatible dataloader")
        # seed = int(checkpoint_prefix_name.split("_")[-1])
        # logging.info(f"Seed extracted from ckpt prefix name: {seed}")
        # datasets = [ds.normal_dataset for ds in alignment_datasets]
        # realignment_dataset = TorchCompatibleIterableDataset(
        #     interleave_datasets(
        #         datasets, seed=seed, probabilities=[1.0 / len(datasets)] * len(datasets)
        #     )
        # )

        # ## make dataloader
        # realignment_dataloader = DataLoader(
        #     realignment_dataset,
        #     shuffle=False,
        #     batch_size=batch_size,
        #     collate_fn=RealignmentAndOtherCollator(
        #         tokenizer,
        #         data_collator,
        #         noaligner=noaligner,
        #     ),
        # )
        # realignment_iterator = iter(realignment_dataloader)

        iterators = [iter(ds) for ds in alignment_datasets]
        realignment_batches = weighted_multilingual_batch(
            probs=uniform_probs,
            noise_mixing_strat=noise_mixing_strat,
            batch_size=batch_size,
            num_batches=realignment_steps,
            iterators=iterators,
            datasets=alignment_datasets,
            tokenizer=tokenizer,
            noaligner=noaligner,
        )
        progress_bar = tqdm(total=realignment_steps)
        with torch.no_grad():
            # for _ in range(realignment_steps):
            for batch in realignment_batches:
                # realignment_iterator, batch, restarted = get_next_or_restart(
                #     realignment_dataloader, realignment_iterator
                # )
                batch = bring_batch_to_model(batch, model)
                outputs = model(**batch, return_dict=True)
                if outputs.per_lang_losses is not None:
                    for lang_id, lang_loss in outputs.per_lang_losses.items():
                        per_lang_loss_sum[lang_id] += lang_loss
                        per_lang_count[lang_id] += 1
                if training_state is not None:
                    training_state.nb_realignment_steps_seen += 1
                progress_bar.update()
        progress_bar.close()

        per_lang_avg = per_lang_loss_sum / per_lang_count.clamp(min=1)
        meta_weight.data.copy_(per_lang_avg)

        logging.info(
            f"Static meta-weights computed from {realignment_steps} batches. "
            f"Per-language avg loss: {per_lang_avg.cpu().tolist()}"
        )

        if checkpoint_path is not None and checkpoint_prefix_name is not None:
            meta_weights_overtime_file = os.path.join(
                checkpoint_path,
                f"{checkpoint_prefix_name}__meta_weights_overtime.jsonl",
            )
            realignment_langs = None
            if result_store:
                realignment_langs = result_store.get_results().get('realignment_langs')
            with open(meta_weights_overtime_file, 'w', encoding='utf-8') as f:
                info = {
                    "realignment_steps": realignment_steps,
                    "batch": realignment_steps,
                    "meta_loss_total": float(per_lang_avg.mean().cpu()),
                    "meta_weight": meta_weight.detach().cpu().tolist(),
                    "realignment_langs": realignment_langs,
                    "realignment_loss": None,
                }
                f.write(json.dumps(info) + "\n")

        return training_state

    if to_train in ["all", "only_meta"]:
        meta_optimizer = torch.optim.Adam([meta_weight], lr=meta_learning_rate)
        meta_optimizer.zero_grad()
    if to_train in ["all", "only_weights"]:
        realignment_optimizer.zero_grad()
        
    progress_bar = tqdm(total=realignment_steps//inner_batches_before_outer)

    iterators = [iter(ds) for ds in alignment_datasets]

    # A file to store meta_weights changes
    meta_weights_overtime_file = os.path.join(checkpoint_path, f"{checkpoint_prefix_name}__meta_weights_overtime.jsonl")
    realignment_langs = None
    if result_store:
        realignment_langs = result_store.get_results().get('realignment_langs')

    with open(meta_weights_overtime_file, 'w', encoding='utf-8') as f:
        for i in range(0, realignment_steps, inner_batches_before_outer):
            meta_loss_total = 0
            probs = torch.softmax(meta_weight / softmax_temp, dim=0)
            # Smoothing epsilon
            probs = (1.0 - lbsmoothing_eps) * probs + (lbsmoothing_eps / probs.size(0))

            realignment_batches = weighted_multilingual_batch(
                probs=probs, 
                noise_mixing_strat=noise_mixing_strat,
                batch_size=batch_size, 
                num_batches=inner_batches_before_outer, 
                iterators=iterators,
                datasets=alignment_datasets,
                tokenizer=tokenizer,
                noaligner=noaligner,
            )

            if i <= start_iter: 
                progress_bar.update()
                continue

            is_finished = False
            for j, batch in enumerate(realignment_batches):

                # ---- Inner loop: realignment batches ----
                realignment_loss = 0

                # Compute realignment loss
                batch = bring_batch_to_model(batch, model)
                outputs = model(**batch, lang_probs=probs, meta_loss_type=meta_loss_type, return_dict=True)

                # realignment_loss += (
                #             realignment_coef / realignment_steps_by_finetuning
                #         ) * outputs.loss
                realignment_loss += outputs.loss
                if to_train in ["all", "only_weights"]:
                    realignment_loss.backward()
                    realignment_optimizer.step()
                    realignment_optimizer.zero_grad()

                meta_loss_total = meta_loss_total + outputs.meta_loss

                if training_state is not None:
                    training_state.nb_realignment_steps_seen += 1

                if isinstance(extra_realignment_steps_checkpoints, list) and i + j + 1 in extra_realignment_steps_checkpoints:
                    save_checkpoint()
                
                # Minus 1 because the first step is 0
                if i + j >= realignment_steps - 1:
                    is_finished = True
                    break
            
            if not is_finished:
                meta_loss_total /= inner_batches_before_outer
                # Regularization to prevent collapse on some languages
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)) if with_regularization else 0
                meta_loss_total -= lambda_entropy * entropy
                # Gradient ascent on meta-loss
                if to_train in ["all", "only_meta"]:
                    (-meta_loss_total).backward()
                    # DEBUG: Similar grad issue
                    # print("meta_weight.grad (mean/std/min/max):",
                    #     meta_weight.grad.mean().item(),
                    #     meta_weight.grad.std().item(),
                    #     meta_weight.grad.min().item(),
                    #     meta_weight.grad.max().item())
                    meta_optimizer.step()
                    meta_optimizer.zero_grad()

            # Write meta_weights changes to file
            meta_loss_detach = meta_loss_total if meta_loss_total == 0  else float(meta_loss_total.detach().cpu())
            meta_weight_detach = meta_weight.detach().cpu().tolist()
            realignment_loss_detach = realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu())
            info = {
                "realignment_steps": realignment_steps,
                "batch": i + j,
                "meta_loss_total": meta_loss_detach,
                "meta_weight": meta_weight_detach,
                "realignment_langs": realignment_langs,
                "realignment_loss": realignment_loss_detach,
            }
            f.write(json.dumps(info) + "\n")

            progress_bar.update()
            if logging_steps is not None and i % logging_steps == 0:
                if training_state is not None:
                    res = training_state.log_state()
                else:
                    logging.info((f"batch: {i}/{realignment_steps} realignment_loss : {realignment_loss_detach} | meta_loss : {meta_loss_detach} {progress_bar}\n"))
                    res = None

                if result_store:
                    result_store.log(
                        {
                            **(res if res is not None else {"realignment_step": i}),
                            "meta_weight": meta_weight_detach,
                            "realignment_loss": realignment_loss_detach,
                            "meta_loss_total": meta_loss_detach,
                        }
                    )
        progress_bar.close()

    # Checkpoint
    if checkpoint_path:
        save_checkpoint()

    return training_state


def ucb_weighted_selection_realignment(
    model,
    tokenizer,
    realignment_optimizer,
    alignment_datasets: list,
    noaligner,
    # UCB-specific parameters
    ucb_exploration_coef=1.0,
    # Shared parameters
    inner_batches_before_outer=5,
    lbsmoothing_eps=0,
    noise_mixing_strat=None,
    softmax_temp=1,
    realignment_steps=1000,
    start_iter=0,
    extra_realignment_steps_checkpoints=None,
    realignment_coef=1.0,
    realignment_steps_by_finetuning=1,
    batch_size=32,
    logging_steps=100,
    result_store=None,
    training_state: Optional[TrainingState] = None,
    checkpoint_path=None,
    checkpoint_prefix_name=None,
    use_adapter=False,
    adapter_approach="",
    adapter_config=None,
):
    '''
    UCB (Upper Confidence Bound) alternative to gradient-based weighted sampling.

    Each language is treated as an arm in a multi-armed bandit. The reward is the
    alignment loss (higher loss = less aligned = should be sampled more). UCB balances
    exploitation (high-loss languages) with exploration (under-sampled languages).

    UCB formula: UCB_i = mean_reward_i + c * sqrt(ln(t) / (n_i + 1))

    UCB scores are converted to sampling probabilities via softmax with temperature.
    '''

    def compute_ucb_probs():
        """Compute sampling probabilities from UCB scores."""
        mean_rewards = cumulative_rewards / pull_counts.clamp(min=1)
        exploration_bonus = ucb_exploration_coef * torch.sqrt(
            torch.log(torch.tensor(max(total_rounds, 1), dtype=torch.float32, device=device))
            / pull_counts.clamp(min=1)
        )
        ucb_scores = mean_rewards + exploration_bonus

        # For languages never pulled, give them high UCB to force exploration
        unpulled = (pull_counts == 0)
        if unpulled.any():
            max_finite = ucb_scores[~unpulled].max() if (~unpulled).any() else torch.tensor(1.0)
            ucb_scores[unpulled] = max_finite + 10.0

        probs = torch.softmax(ucb_scores / softmax_temp, dim=0)
        # Label smoothing
        if lbsmoothing_eps > 0:
            probs = (1.0 - lbsmoothing_eps) * probs + (lbsmoothing_eps / num_langs)
        return probs, ucb_scores

    def save_checkpoint():
        realignment_iter = i + j
        ckpt_file_name = f"{checkpoint_prefix_name}_iter_{realignment_iter}.ckpt"
        logging.info(f"Saved model at {realignment_iter}/{realignment_steps} at {ckpt_file_name}")
        model_state_dict = extract_adapter_state_dict(model, use_adapter, adapter_approach)
        torch.save({
            'iter': realignment_iter,
            'nb_iter': realignment_steps,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': realignment_optimizer.state_dict(),
            'loss': realignment_loss if realignment_loss == 0 else float(realignment_loss.detach().cpu()),
            "ucb_scores": ucb_scores_detach,
            "pull_counts": pull_counts.cpu().tolist(),
            "cumulative_rewards": cumulative_rewards.cpu().tolist(),
            'use_adapter': use_adapter,
            'adapter_approach': adapter_approach,
            'adapter_config': adapter_config,
        }, os.path.join(checkpoint_path, ckpt_file_name))

    if not realignment_steps:
        raise Exception("realignment_steps must be provided")

    device = model.device

    num_langs = len(alignment_datasets)
    # UCB statistics
    cumulative_rewards = torch.zeros(num_langs, device=device)
    pull_counts = torch.zeros(num_langs, device=device)
    total_rounds = 0

    realignment_optimizer.zero_grad()

    progress_bar = tqdm(total=realignment_steps // inner_batches_before_outer)

    iterators = [iter(ds) for ds in alignment_datasets]

    # File to store UCB weights changes
    ucb_weights_overtime_file = os.path.join(checkpoint_path, f"{checkpoint_prefix_name}__ucb_weights_overtime.jsonl")
    realignment_langs = None
    if result_store:
        realignment_langs = result_store.get_results().get('realignment_langs')

    with open(ucb_weights_overtime_file, 'w', encoding='utf-8') as f:
        for i in range(0, realignment_steps, inner_batches_before_outer):
            # 1. Compute UCB scores -> probabilities
            probs, ucb_scores = compute_ucb_probs()

            # 2. Sample batches using probabilities
            realignment_batches = weighted_multilingual_batch(
                probs=probs,
                noise_mixing_strat=noise_mixing_strat,
                batch_size=batch_size,
                num_batches=inner_batches_before_outer,
                iterators=iterators,
                datasets=alignment_datasets,
                tokenizer=tokenizer,
                noaligner=noaligner,
                device=device,
            )

            if i <= start_iter: 
                progress_bar.update()
                continue

            is_finished = False
            for j, batch in enumerate(realignment_batches):

                # ---- Inner loop: realignment batches ----
                realignment_loss = 0

                batch = bring_batch_to_model(batch, model)
                # Do NOT pass lang_probs (no gradient-based meta-loss needed)
                outputs = model(**batch, return_dict=True)

                realignment_loss += outputs.loss
                realignment_loss.backward()
                realignment_optimizer.step()
                realignment_optimizer.zero_grad()

                # Update UCB statistics from per-language losses
                if outputs.per_lang_losses is not None:
                    for lang_id, lang_loss in outputs.per_lang_losses.items():
                        cumulative_rewards[lang_id] += lang_loss
                        pull_counts[lang_id] += 1
                    total_rounds += 1

                if training_state is not None:
                    training_state.nb_realignment_steps_seen += 1

                if isinstance(extra_realignment_steps_checkpoints, list) and i + j + 1 in extra_realignment_steps_checkpoints:
                    save_checkpoint()

                if i + j >= realignment_steps - 1:
                    is_finished = True
                    break

            # Log UCB weights
            ucb_scores_detach = ucb_scores.detach().cpu().tolist()
            probs_detach = probs.detach().cpu().tolist()
            realignment_loss_detach = realignment_loss if realignment_loss == 0 else float(realignment_loss.detach().cpu())
            mean_rewards_detach = (cumulative_rewards / pull_counts.clamp(min=1)).cpu().tolist()
            info = {
                "realignment_steps": realignment_steps,
                "batch": i + j,
                "ucb_scores": ucb_scores_detach,
                "sampling_probs": probs_detach,
                "pull_counts": pull_counts.cpu().tolist(),
                "mean_rewards": mean_rewards_detach,
                "realignment_langs": realignment_langs,
                "realignment_loss": realignment_loss_detach,
            }
            f.write(json.dumps(info) + "\n")

            progress_bar.update()
            if logging_steps is not None and i % logging_steps == 0:
                if training_state is not None:
                    res = training_state.log_state()
                else:
                    logging.info(f"batch: {i}/{realignment_steps} realignment_loss: {realignment_loss_detach} | pull_counts: {pull_counts.cpu().tolist()} {progress_bar}\n")
                    res = None

                if result_store:
                    result_store.log(
                        {
                            **(res if res is not None else {"realignment_step": i}),
                            "ucb_scores": ucb_scores_detach,
                            "sampling_probs": probs_detach,
                            "realignment_loss": realignment_loss_detach,
                            "mean_rewards": mean_rewards_detach,
                        }
                    )
        progress_bar.close()

    # Checkpoint
    if checkpoint_path:
        save_checkpoint()

    return training_state