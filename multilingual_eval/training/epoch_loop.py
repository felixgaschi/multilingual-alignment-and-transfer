import os
import itertools
import torch
import logging
import math
from typing import Optional
import random
import numpy as np
from tqdm import tqdm
from transformers.optimization import get_scheduler

from multilingual_eval.training.utils import bring_batch_to_model, get_next_or_restart
from multilingual_eval.training.states import TrainingState

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
        if scheduling_freeze and phase and direction and i > 0 and i % (nb_iter // progress) == 0:
            # Checkpoint
            if checkpoint_path:
                logging.info(f"Saved model at {i}/{nb_iter} at {checkpoint_prefix_name}_iter_{i}.ckpt")
                torch.save({
                    'iter': i,
                    'nb_iter': nb_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu()),
                }, os.path.join(checkpoint_path, f"{checkpoint_prefix_name}_iter_{i}.ckpt"))
            
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
    if checkpoint_path:
        logging.info(f"Saved model at {i}/{nb_iter} at {checkpoint_prefix_name}_iter_{i}.ckpt")
        torch.save({
            'iter': i,
            'nb_iter': nb_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': realignment_loss if realignment_loss == 0  else float(realignment_loss.detach().cpu()),
        }, os.path.join(checkpoint_path, f"{checkpoint_prefix_name}_iter_{i}.ckpt"))

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
