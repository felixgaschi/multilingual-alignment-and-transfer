from datasets.iterable_dataset import IterableDataset
import logging
from typing import Optional, Any, Callable, List
from dataclasses import dataclass, field
import torch
from copy import deepcopy
from transformers.optimization import get_scheduler

from transformers import DataCollatorForTokenClassification, DataCollator


from multilingual_eval.training.alignment_evaluation_loops import evaluate_alignment
from multilingual_eval.utils import get_nb_layers
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentCollator,
)
from multilingual_eval.training.epoch_loop import fine_tuning_loop, realignment_epoch
from multilingual_eval.training.evaluation_loops import evaluate_any_task
from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset


@dataclass
class FineTuningTask:
    train_dataset: torch.utils.data.Dataset
    eval_datasets: List[torch.utils.data.Dataset]
    eval_datasets_prefixes: List[str]
    task_name: str
    data_collator: DataCollator
    metric_fn: Callable
    model_fn: Callable
    learning_rate: float
    batch_size: int = 4
    accumulation_steps: int = 8
    remove_from_input_for_eval: List[str] = field(default_factory=lambda: [])
    keep_in_input_for_eval: List[str] = field(default_factory=lambda: ["labels"])
    keep_in_output_for_eval: List[str] = field(default_factory=lambda: ["logits"])


def fine_tune_and_evaluate(
    model, task: FineTuningTask, seed=None, fine_tuning_steps=2000, log_fn=None
):
    log_fn = log_fn or logging.info

    fine_tuning_loop(
        model,
        task.train_dataset,
        task.data_collator,
        task.task_name,
        batch_size=task.batch_size,
        accumulation_steps=task.accumulation_steps,
        seed=seed,
        steps=fine_tuning_steps,
        learning_rate=task.learning_rate,
    )

    for prefix, dataset in zip(task.eval_datasets_prefixes, task.eval_datasets):
        res = evaluate_any_task(
            model,
            torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                batch_size=task.batch_size,
                collate_fn=task.data_collator,
            ),
            task.metric_fn,
            prefix=f"{task.task_name}_eval_{prefix}",
            remove_from_input=task.remove_from_input_for_eval,
            keep_in_input=task.keep_in_input_for_eval,
            keep_in_output=task.keep_in_output_for_eval,
        )

        log_fn(res)


def evaluate_model_for_alignment_and_generalizationt(
    tokenizer,
    model,
    model_name: str,
    fine_tuning_tasks: List[FineTuningTask],
    eval_realignment_dataset,
    device="cpu",
    seed=None,
    fine_tuning_steps=2000,
    log_fn=None,
    strong_alignment=True,
    nb_pairs=5000,
):
    log_fn = log_fn or logging.info

    n_layers = get_nb_layers(model)

    for task in fine_tuning_tasks:
        model_to_finetune = task.model_fn(model_name)

        finetuning_state_dict = model_to_finetune.state_dict()

        new_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if k in finetuning_state_dict and v.size() == finetuning_state_dict[k].size()
        }

        model_to_finetune.load_state_dict(new_state_dict, strict=False)

        model_to_finetune.to(device)
        fine_tune_and_evaluate(
            model_to_finetune, task, seed=seed, fine_tuning_steps=fine_tuning_steps, log_fn=log_fn
        )

        scores_fwd, scores_bwd = evaluate_alignment(
            tokenizer,
            model_to_finetune,
            eval_realignment_dataset,
            nb_pairs=nb_pairs,
            strong_alignment=strong_alignment,
            layers=list(range(n_layers)),
        )

        res = {
            **{f"alignment_after_{task.task_name}_fwd_{i}": s for i, s in enumerate(scores_fwd)},
            **{f"alignment_after_{task.task_name}_bwd_{i}": s for i, s in enumerate(scores_bwd)},
        }
        log_fn(res)

        del model_to_finetune

    if eval_realignment_dataset is not None:
        model.to(device)
        scores_fwd, scores_bwd = evaluate_alignment(
            tokenizer,
            model,
            eval_realignment_dataset,
            nb_pairs=nb_pairs,
            strong_alignment=strong_alignment,
            layers=list(range(n_layers)),
        )
        model.to("cpu")

        res = {
            **{f"alignment_before_fwd_{i}": s for i, s in enumerate(scores_fwd)},
            **{f"alignment_before_bwd_{i}": s for i, s in enumerate(scores_bwd)},
        }
        log_fn(res)


def realignment_loop(
    tokenizer,
    model,
    model_name,
    realignment_dataset: IterableDataset,
    realignment_batch_size=16,
    realignment_steps_by_epochs=2_000,
    realignment_learning_rate=2e-5,
    realignment_epochs=10,
    fine_tuning_tasks: List[FineTuningTask] = None,
    fine_tuning_steps=2_000,
    nb_pairs=5000,
    strong_alignment=True,
    noaligner=False,
    log_in_wandb=False,
    seed=None,
    device="cuda:0" if torch.cuda.device_count() > 0 else "cpu",
):
    if log_in_wandb:
        import wandb

        def log_fn(res):
            logging.info(res)
            wandb.log(res)

    else:
        log_fn = logging.info

    if nb_pairs == 0:
        eval_realignment_dataset = None
        train_realignment_dataset = TorchCompatibleIterableDataset(realignment_dataset)
    else:
        eval_realignment_dataset = TorchCompatibleIterableDataset(realignment_dataset.take(nb_pairs))
        train_realignment_dataset = TorchCompatibleIterableDataset(realignment_dataset.skip(nb_pairs))

    fine_tuning_tasks = fine_tuning_tasks or []

    evaluate_model_for_alignment_and_generalizationt(
        tokenizer,
        model,
        model_name,
        fine_tuning_tasks,
        eval_realignment_dataset,
        device=device,
        seed=seed,
        fine_tuning_steps=fine_tuning_steps,
        log_fn=log_fn,
        strong_alignment=strong_alignment,
        nb_pairs=nb_pairs,
    )

    realignment_dataloader = torch.utils.data.DataLoader(
        train_realignment_dataset,
        shuffle=False,
        batch_size=realignment_batch_size,
        collate_fn=RealignmentCollator(tokenizer, noaligner=noaligner),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=realignment_learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * realignment_steps_by_epochs * realignment_epochs),
        num_training_steps=realignment_steps_by_epochs * realignment_epochs,
    )
    iterator = iter(realignment_dataloader)
    for i_epoch in range(realignment_epochs):
        model.to(device)
        realignment_epoch(
            model,
            iterator,
            realignment_dataloader,
            optimizer,
            scheduler=scheduler,
            batch_size=realignment_batch_size,
            steps=realignment_steps_by_epochs,
        )
        model.to("cpu")

        evaluate_model_for_alignment_and_generalizationt(
            tokenizer,
            model,
            model_name,
            fine_tuning_tasks,
            eval_realignment_dataset,
            device=device,
            seed=seed,
            fine_tuning_steps=fine_tuning_steps,
            log_fn=log_fn,
            strong_alignment=strong_alignment,
            nb_pairs=nb_pairs,
        )
