import logging
import json
from typing import Optional

import torch

from multilingual_eval.training.states import TrainingState
from multilingual_eval.training.epoch_loop import weighted_selection_realignment


def load_meta_weight_from_jsonl(path, device, expected_num_langs=None):
    last_weight = None
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            data = json.loads(line)
            if "meta_weight" in data:
                last_weight = data["meta_weight"]

    if last_weight is None:
        raise ValueError(f"No meta_weight entry found in {path}")
    if expected_num_langs is not None and len(last_weight) != expected_num_langs:
        raise ValueError(
            f"Loaded {len(last_weight)} meta-weights from {path}, "
            f"but expected {expected_num_langs}"
        )

    return torch.nn.Parameter(
        torch.tensor(last_weight, dtype=torch.float32, device=device)
    )


def decoupled_weighted_selection_realignment(
    model,
    tokenizer,
    realignment_optimizer,
    alignment_datasets: list,
    noaligner,
    meta_weight=None,
    closed_form_meta=False,
    static_meta_weights_jsonl=None,
    **kwargs,
):
    """
    Two-phase wrapper around weighted_selection_realignment:

    Phase 1 — learn sampling weights. Two variants:
        - gradient-based (default, closed_form_meta=False): freeze all model
          parameters, call weighted_selection_realignment so only the language
          meta-weights are updated via gradient ascent on the meta-loss.
        - closed-form (closed_form_meta=True): run one pass over the data with
          uniform sampling, average per-language realignment losses (no
          backprop, no optimizer), and set meta_weight so softmax(meta_weight)
          is the desired sampling distribution.

    Phase 2 — train the model: restore model parameters, freeze the meta-weights,
    call weighted_selection_realignment again so the model is updated with the
    now-static sampling distribution learned in phase 1.
    """
    if static_meta_weights_jsonl is not None:
        meta_weight = load_meta_weight_from_jsonl(
            static_meta_weights_jsonl,
            device="cuda",
            expected_num_langs=len(alignment_datasets),
        )
        logging.info(
            "Loaded static meta-weights from "
            f"{static_meta_weights_jsonl}; skipping decoupled phase 1."
        )
    elif meta_weight is None:
        meta_weight = torch.nn.Parameter(torch.ones(len(alignment_datasets), device="cuda"))

    # ---- Phase 1: learn meta-weights ----
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    if static_meta_weights_jsonl is None:
        if closed_form_meta:
            logging.info(
                "Decoupled realignment phase 1: computing meta-weights from "
                "per-language losses (closed form)."
            )
            phase1_to_train = "only_meta_closed"
        else:
            logging.info(
                "Decoupled realignment phase 1: learning meta-weights with model frozen."
            )
            for param in model.parameters():
                param.requires_grad = False
            phase1_to_train = "only_meta"

        training_state = weighted_selection_realignment(
            model=model,
            tokenizer=tokenizer,
            realignment_optimizer=realignment_optimizer,
            alignment_datasets=alignment_datasets,
            noaligner=noaligner,
            meta_weight=meta_weight,
            to_train=phase1_to_train,
            **kwargs,
        )
    else:
        training_state = kwargs.get("training_state")

    # ---- Phase 2: train model with frozen meta-weights ----
    logging.info("Decoupled realignment phase 2: training model with static meta-weights.")
    for param, was_trainable in zip(model.parameters(), original_requires_grad):
        param.requires_grad = was_trainable
    meta_weight.requires_grad = False

    if "training_state" in kwargs:
        del kwargs["training_state"]

    training_state = weighted_selection_realignment(
        model=model,
        tokenizer=tokenizer,
        realignment_optimizer=realignment_optimizer,
        alignment_datasets=alignment_datasets,
        noaligner=noaligner,
        meta_weight=meta_weight,
        training_state=training_state,
        to_train="only_weights",
        **kwargs,
    )

    meta_weight.requires_grad = True
    return training_state
