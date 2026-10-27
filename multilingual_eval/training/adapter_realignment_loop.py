import os
import torch
import random
import logging
import itertools
import numpy as np

from tqdm import tqdm
from transformers import set_seed, AdapterSetup
from transformers.optimization import AdamW


from multilingual_eval.training.utils import bring_batch_to_model
from multilingual_eval.datasets.realignment_dataset import RealignmentCollator
from multilingual_eval.models.realignment_loss import compute_realignment_loss


def realignment_loop_with_adapters(
    tokenizer,
    model,
    realignment_datasets,
    pairs,
    probabilities=None,
    n_steps=1_000_000,
    realignment_batch_size=16,
    learning_rate=1e-4,
    logging_steps=None,
    log_in_wandb=False,
    seed=None,
    num_workers=0,
    strong_alignment=True,
    no_adapter_for=None,
):
    assert len(pairs) == len(realignment_datasets)

    data_collator = RealignmentCollator(tokenizer)
    probabilities = probabilities or [1 / len(realignment_datasets)] * len(realignment_datasets)
    logging_steps = logging_steps or n_steps // 100

    if log_in_wandb:
        import wandb

    # Put model to GPU if available
    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # Fix random seed for Pytorch and numpy
    set_seed(seed)

    dataloaders = [
        torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            shuffle=False,
            batch_size=realignment_batch_size,
            collate_fn=data_collator,
        )
        for dataset in realignment_datasets
    ]

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    iterators = list(map(iter, dataloaders))

    progress_bar = tqdm(total=n_steps, file=open(os.devnull, "w"))

    realignment_transformation = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(model.config.hidden_size, 128, bias=False),
    ).to(model.device)

    model.train_adapter(
        [
            f"{lang}_adapter"
            for lang in itertools.chain(map(lambda x: x[0], pairs), map(lambda x: x[1], pairs))
            if lang != no_adapter_for
        ]
    )

    for i_step in range(n_steps):

        i_iterator = np.random.choice(list(range(len(iterators))), p=probabilities)
        left, right = pairs[i_iterator]

        try:
            batch = next(iterators[i_iterator])
        except StopIteration:
            logging.warning(
                f"Reached end of {i_iterator}-th realignment iterator for pair: {left}, {right}"
            )
            iterators[i_iterator] = iter(dataloaders[i_iterator])
            batch = next(iterators[i_iterator])

        batch = bring_batch_to_model(batch, model)

        loss, _, _ = compute_realignment_loss(
            model,  # getattr(model, model.base_model_prefix),
            realignment_transformation,
            [-1],
            strong_alignment=strong_alignment,
            left_context=(AdapterSetup(f"{left}_adapter") if (left != no_adapter_for) else None),
            right_context=(AdapterSetup(f"{right}_adapter") if (right != no_adapter_for) else None),
            **batch,
        )

        loss.backward()

        optimizer.step()

        progress_bar.update()

        if i_step % logging_steps == 0:

            logging.info(f"loss: {loss}. {progress_bar}")

            if log_in_wandb:
                wandb.log({"realignment_loss": loss, "realignment_step": i_step})

    progress_bar.close()
