from collections import defaultdict
from typing import Dict
from multilingual_eval.training.utils import bring_batch_to_model, prefix_dictionary

import logging

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, QuestionAnsweringPipeline
from transformers.trainer_pt_utils import nested_concat

from multilingual_eval.datasets.token_classification import get_token_classification_metrics
from multilingual_eval.datasets.xquad import get_xquad
from multilingual_eval.datasets.xlsum import (
    get_xlsum,
    compute_rouge,
    build_xlsum_clm_prompt,
)

from multilingual_eval.models.simplified import SimplifiedModelForRealignment


def evaluate_any_task(
    model,
    eval_dataloader,
    metric_fn,
    prefix="eval",
    remove_from_input=None,
    keep_in_input=None,
    keep_in_output=None,
    padding_index=-100,
):
    remove_from_input = remove_from_input or []
    keep_in_input = keep_in_input or ["labels"]
    keep_in_output = keep_in_output or ["logits"]

    model.eval()

    all_results = {key: None for key in keep_in_input + keep_in_output}
    for i, batch in enumerate(eval_dataloader):
        results = {key: batch[key].numpy() for key in keep_in_input}
        for key in remove_from_input:
            del batch[key]
        batch = bring_batch_to_model(batch, model)

        outputs = model(**batch, return_dict=True)

        results.update({key: outputs[key].detach().cpu().numpy() for key in keep_in_output})

        all_results = {
            key: results[key]
            if val is None
            else nested_concat(val, results[key], padding_index=padding_index)
            for key, val in all_results.items()
        }

    return prefix_dictionary(metric_fn(all_results), prefix)


def evaluate_token_classification(
    model, eval_dataloader, prefix="eval", metric_fn=None, label_key="labels"
):
    """
    Evaluates a model on a given dataloader
    """
    model.eval()

    metric_fn = metric_fn or get_token_classification_metrics()

    all_labels = None
    all_predictions = None
    for i, batch in enumerate(eval_dataloader):
        labels = batch[label_key].numpy()
        batch = bring_batch_to_model(batch, model)

        predictions = model(**batch, return_dict=True).logits
        predictions = predictions.detach().cpu().numpy()

        all_labels = (
            labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        )
        all_predictions = (
            predictions
            if all_predictions is None
            else nested_concat(all_predictions, predictions, padding_index=-100)
        )

    return prefix_dictionary(metric_fn((all_predictions, all_labels)), prefix)


def evaluate_several_token_classification(
    tokenizer,
    model,
    datasets,
    batch_size,
    prefixes=None,
    overall_prefix=None,
    metric_fn=None,
    collator=None,
    label_key="labels",
):
    """
    Evaluates a model on several datasets, also aggregates the metrics with prefix "avg".
    Metrics will have prefixes of the form {overall_prefix}_{prefixes[i] or avg}_name_of_the_metrics
    """
    collator = collator or DataCollatorForTokenClassification(tokenizer)
    prefixes = prefixes or [str(i) for i in range(len(datasets))]
    assert len(datasets) == len(prefixes)
    assert "avg" not in prefixes

    dataloaders = [
        DataLoader(dataset, batch_size=batch_size, collate_fn=collator) for dataset in datasets
    ]
    res = {}
    agg = defaultdict(lambda: 0)
    for dataloader, prefix in zip(dataloaders, prefixes):
        next_res = evaluate_token_classification(
            model, dataloader, prefix=None, metric_fn=metric_fn, label_key=label_key
        )
        for key, value in next_res.items():
            agg[key] += value
        res.update(prefix_dictionary(next_res, prefix))

    res.update(prefix_dictionary({k: v / len(datasets) for k, v in agg.items()}, prefix="avg"))

    return prefix_dictionary(res, overall_prefix)


def evaluate_xquad(
    model,
    tokenizer,
    left_lang,
    right_langs,
    batch_size=16,
    debug=False,
    data_cache_dir=None,
    log_in_wandb=False,
    result_store=None,
):
    if isinstance(model, SimplifiedModelForRealignment):
        model = model.model

    oracle = QuestionAnsweringPipeline(
        model=model, tokenizer=tokenizer, device=model.device.index, batch_size=batch_size
    )

    validation_datasets = list(
        map(
            list,
            get_xquad(
                right_langs,
                tokenizer,
                split="test",
                limit=100 if debug else None,
                datasets_cache_dir=data_cache_dir,
                interleave=False,
                preprocessing=False,
            ),
        )
    )

    source_validation_dataset = list(
        get_xquad(
            left_lang,
            tokenizer,
            split="test",
            limit=100 if debug else None,
            datasets_cache_dir=data_cache_dir,
            preprocessing=False,
        )
    )

    def normalize_text(s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(prediction, truth):
        pred_tokens = normalize_text(prediction).split()
        truth_tokens = normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / max(1, len(pred_tokens))
        rec = len(common_tokens) / max(1, len(truth_tokens))

        return 2 * (prec * rec) / max(1, prec + rec)

    res = {}
    agg = defaultdict(lambda: 0)
    for lang, examples in zip(right_langs, validation_datasets):
        predictions = oracle(
            list(map(lambda y: {"question": y["question"], "context": y["context"]}, examples))
        )

        predictions = list(map(lambda x: x["answer"], predictions))
        references = list(map(lambda x: x["answers"]["text"][0], examples))

        results = {
            "em": sum([int(x == y) for x, y in zip(predictions, references)]) / len(predictions),
            "f1": sum([compute_f1(x, y) for x, y in zip(predictions, references)])
            / len(predictions),
        }

        for key, value in results.items():
            agg[key] += value
        res.update(prefix_dictionary(results, lang))

    res.update(
        prefix_dictionary({k: v / len(validation_datasets) for k, v in agg.items()}, prefix="avg")
    )
    res = prefix_dictionary(res, "eval")

    predictions = oracle(
        list(
            map(
                lambda y: {"question": y["question"], "context": y["context"]},
                source_validation_dataset,
            )
        )
    )
    predictions = list(map(lambda x: x["answer"], predictions))
    references = list(map(lambda x: x["answers"]["text"][0], source_validation_dataset))

    results = {
        "em": sum(
            [int(normalize_text(x) == normalize_text(y)) for x, y in zip(predictions, references)]
        )
        / len(predictions),
        "f1": sum([compute_f1(x, y) for x, y in zip(predictions, references)]) / len(predictions),
    }
    res.update(prefix_dictionary(results, "eval_same"))

    logging.info(res)
    if log_in_wandb:
        import wandb

        wandb.log(res)
    if result_store:
        result_store.log(res)


def evaluate_xlsum(
    model,
    tokenizer,
    left_lang,
    right_langs,
    batch_size=16,
    debug=False,
    data_cache_dir=None,
    log_in_wandb=False,
    result_store=None,
    max_length=512,
    max_target_length=64,
    num_beams=1,
):
    """
    Evaluates a seq2seq or decoder-only model on XLSum with generation,
    computing ROUGE for each target language and the source language
    """
    import torch

    if isinstance(model, SimplifiedModelForRealignment):
        model = model.model

    model.eval()

    is_seq2seq = getattr(model.config, "is_encoder_decoder", False)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def generate_seq2seq_batch(texts):
        inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_target_length, num_beams=num_beams
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def generate_causal_batch(texts):
        # left-pad the prompts so that generated tokens all start at the
        # same position, then strip the prompt before decoding
        prompts = [
            build_xlsum_clm_prompt(
                tokenizer, text, max_length=max_length, max_target_length=max_target_length
            )
            for text in texts
        ]
        longest = max(map(len, prompts))
        input_ids = torch.tensor(
            [[pad_token_id] * (longest - len(p)) + p for p in prompts]
        ).to(model.device)
        attention_mask = torch.tensor(
            [[0] * (longest - len(p)) + [1] * len(p) for p in prompts]
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_target_length,
                num_beams=num_beams,
                pad_token_id=pad_token_id,
            )
        return tokenizer.batch_decode(generated[:, longest:], skip_special_tokens=True)

    def generate_summaries(examples, lang):
        predictions = []
        texts = examples["text"]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if is_seq2seq:
                predictions.extend(generate_seq2seq_batch(batch))
            else:
                predictions.extend(generate_causal_batch(batch))
        return compute_rouge(predictions, examples["summary"], lang=lang)

    validation_datasets = get_xlsum(
        right_langs,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
        interleave=False,
        preprocessing=False,
    )
    if not isinstance(validation_datasets, list):
        validation_datasets = [validation_datasets]

    res = {}
    agg = defaultdict(lambda: 0)
    for lang, examples in zip(right_langs, validation_datasets):
        results = generate_summaries(examples, lang)

        for key, value in results.items():
            agg[key] += value
        res.update(prefix_dictionary(results, lang))

    res.update(
        prefix_dictionary({k: v / len(validation_datasets) for k, v in agg.items()}, prefix="avg")
    )
    res = prefix_dictionary(res, "eval")

    source_validation_dataset = get_xlsum(
        left_lang,
        tokenizer,
        split="test",
        limit=100 if debug else None,
        datasets_cache_dir=data_cache_dir,
        preprocessing=False,
    )
    res.update(prefix_dictionary(generate_summaries(source_validation_dataset, left_lang), "eval_same"))

    logging.info(res)
    if log_in_wandb:
        import wandb

        wandb.log(res)
    if result_store:
        result_store.log(res)
