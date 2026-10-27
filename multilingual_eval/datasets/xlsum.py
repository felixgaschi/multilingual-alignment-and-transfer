from typing import Union, List
from collections import Counter

import numpy as np
from datasets import interleave_datasets

from multilingual_eval.datasets.data_utils import (
    convert_dataset_to_iterable_dataset,
    load_dataset_cached,
)


# Mapping from language codes to the 45 configurations of csebuetnlp/xlsum
# (maximum coverage of the dataset)
XLSUM_LANG_TO_CONFIG = {
    "am": "amharic",
    "ar": "arabic",
    "az": "azerbaijani",
    "bn": "bengali",
    "my": "burmese",
    "zh": "chinese_simplified",
    "zh_trad": "chinese_traditional",
    "en": "english",
    "fr": "french",
    "gu": "gujarati",
    "ha": "hausa",
    "hi": "hindi",
    "ig": "igbo",
    "id": "indonesian",
    "ja": "japanese",
    "rn": "kirundi",
    "ko": "korean",
    "ky": "kyrgyz",
    "mr": "marathi",
    "ne": "nepali",
    "om": "oromo",
    "ps": "pashto",
    "fa": "persian",
    "pcm": "pidgin",
    "pt": "portuguese",
    "pa": "punjabi",
    "ru": "russian",
    "gd": "scottish_gaelic",
    "sr": "serbian_cyrillic",
    "sr_latn": "serbian_latin",
    "si": "sinhala",
    "so": "somali",
    "es": "spanish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "ti": "tigrinya",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "uz": "uzbek",
    "vi": "vietnamese",
    "cy": "welsh",
    "yo": "yoruba",
}

# Languages written without whitespace between words, for which
# ROUGE is computed at the character level
XLSUM_NON_WHITESPACE_LANGS = {
    "zh",
    "zh_trad",
    "chinese_simplified",
    "chinese_traditional",
    "ja",
    "japanese",
    "my",
    "burmese",
    "th",
    "thai",
}


def xlsum_config_name(lang: str):
    """
    Resolve a language code (or an already valid configuration name)
    to a configuration name of csebuetnlp/xlsum
    """
    config = XLSUM_LANG_TO_CONFIG.get(lang, lang)
    if config not in XLSUM_LANG_TO_CONFIG.values():
        raise KeyError(
            f"Language `{lang}` is not available in XLSum. "
            f"Available codes: {sorted(XLSUM_LANG_TO_CONFIG)}"
        )
    return config


class XLSumMapper:
    def __init__(self, tokenizer, max_length=None, max_target_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
        res = self.tokenizer(
            examples["text"], max_length=self.max_length, truncation=True
        )
        labels = self.tokenizer(
            text_target=examples["summary"],
            max_length=self.max_target_length,
            truncation=True,
        )
        res["labels"] = labels["input_ids"]
        return res


# Cue appended to the article when fine-tuning decoder-only models,
# must be identical at training and generation time
XLSUM_CLM_SEPARATOR = "\nSummary:"


def build_xlsum_clm_prompt(tokenizer, text, max_length=512, max_target_length=64):
    """
    Build the prompt token ids (article + separator) for decoder-only
    summarization, keeping room for max_target_length summary tokens
    within max_length
    """
    sep_ids = tokenizer(XLSUM_CLM_SEPARATOR, add_special_tokens=False)["input_ids"]
    budget = max(max_length - max_target_length - len(sep_ids) - 2, 1)
    text_ids = tokenizer(
        text, truncation=True, max_length=budget, add_special_tokens=False
    )["input_ids"]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return bos + text_ids + sep_ids


class XLSumCausalMapper:
    """
    Packs article + separator + summary into a single sequence for
    decoder-only models, with the loss masked on the prompt part
    """

    def __init__(self, tokenizer, max_length=None, max_target_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for text, summary in zip(examples["text"], examples["summary"]):
            prompt_ids = build_xlsum_clm_prompt(
                self.tokenizer,
                text,
                max_length=self.max_length,
                max_target_length=self.max_target_length,
            )
            summary_ids = self.tokenizer(
                summary,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=False,
            )["input_ids"]
            if self.tokenizer.eos_token_id is not None:
                summary_ids = summary_ids + [self.tokenizer.eos_token_id]

            input_ids = prompt_ids + summary_ids
            all_input_ids.append(input_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_labels.append([-100] * len(prompt_ids) + summary_ids)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }


def get_xlsum(
    lang: Union[List[str], str],
    tokenizer,
    limit=None,
    split="train",
    datasets_cache_dir=None,
    interleave=True,
    lang_id=None,
    return_length=False,
    n_epochs=1,
    remove_useless=True,
    max_length=512,
    max_target_length=64,
    preprocessing=True,
    causal=False,
):
    """
    Return XLSum dataset (csebuetnlp/xlsum) for summarization,
    tokenized for a seq2seq model (article -> summary), or for a
    decoder-only model if causal=True, unless preprocessing=False
    """
    if not isinstance(lang, list):
        lang = [lang]
    if lang_id is not None:
        if not isinstance(lang_id, list):
            lang_id = [lang_id]
        assert len(lang_id) == len(lang)

    configs = [xlsum_config_name(elt) for elt in lang]

    datasets = [
        load_dataset_cached("csebuetnlp/xlsum", config, cache_dir=datasets_cache_dir)[split]
        for config in configs
    ]

    n_datasets = len(datasets)

    if limit:
        limits = [
            limit // n_datasets + (1 if i < limit % n_datasets else 0) for i in range(n_datasets)
        ]

        datasets = list(
            map(
                lambda x: x[0].shuffle().filter(lambda _, i: i < x[1], with_indices=True),
                zip(datasets, limits),
            )
        )

    if n_datasets > 1 and interleave:
        datasets = [interleave_datasets(datasets)]

    if return_length:
        lengths = list(map(len, datasets))

    if not preprocessing:
        if n_datasets == 1 or interleave:
            if return_length:
                return datasets[0], lengths[0]
            return datasets[0]
        if return_length:
            return datasets, lengths
        return datasets

    if n_epochs > 1:
        datasets = map(lambda x: convert_dataset_to_iterable_dataset(x, n_epochs), datasets)

    mapper_class = XLSumCausalMapper if causal else XLSumMapper
    datasets = list(
        map(
            lambda x: x.map(
                mapper_class(
                    tokenizer,
                    max_length=max_length,
                    max_target_length=max_target_length,
                ),
                batched=True,
            ),
            datasets,
        ),
    )

    if lang_id is not None:
        datasets = list(
            map(lambda x: x[0].map(lambda y: {**y, "lang_id": [x[1]]}), zip(datasets, lang_id))
        )

    if remove_useless:
        datasets = list(
            map(
                lambda x: x.remove_columns(["id", "url", "title", "summary", "text"]),
                datasets,
            )
        )

    if n_datasets == 1 or interleave:
        if return_length:
            return datasets[0], lengths[0]
        return datasets[0]
    if return_length:
        return datasets, lengths
    return datasets


def get_xlsum_clm(*args, **kwargs):
    """
    XLSum tokenized for decoder-only models (article + separator + summary
    in one sequence, loss masked on the prompt)
    """
    return get_xlsum(*args, causal=True, **kwargs)


def tokenize_for_rouge(text: str, lang=None):
    """
    Language-agnostic tokenization for ROUGE: lowercased whitespace tokens,
    or characters for languages written without whitespace
    """
    if lang in XLSUM_NON_WHITESPACE_LANGS:
        return [c for c in text.strip() if not c.isspace()]
    return text.lower().split()


def _fmeasure(overlap, pred_count, ref_count):
    if pred_count == 0 or ref_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_count
    recall = overlap / ref_count
    return 2 * precision * recall / (precision + recall)


def rouge_n(pred_tokens, ref_tokens, n=1):
    pred_ngrams = Counter(zip(*[pred_tokens[i:] for i in range(n)]))
    ref_ngrams = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return _fmeasure(overlap, max(len(pred_tokens) - n + 1, 0), max(len(ref_tokens) - n + 1, 0))


def rouge_l(pred_tokens, ref_tokens):
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    # LCS length by dynamic programming, single-row
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for j, ref_token in enumerate(ref_tokens):
            if pred_token == ref_token:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[-1]))
        previous = current
    lcs = previous[-1]
    return _fmeasure(lcs, len(pred_tokens), len(ref_tokens))


def compute_rouge(predictions: List[str], references: List[str], lang=None):
    """
    Compute average ROUGE-1, ROUGE-2 and ROUGE-L F1 scores with an in-repo
    implementation (whitespace tokenization, character-level for zh/ja/my/th)

    Note: unlike the official XL-Sum evaluation, this does not use
    language-specific stemmers/tokenizers, so scores are indicative
    """
    assert len(predictions) == len(references)
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_for_rouge(pred, lang=lang)
        ref_tokens = tokenize_for_rouge(ref, lang=lang)
        scores["rouge1"] += rouge_n(pred_tokens, ref_tokens, n=1)
        scores["rouge2"] += rouge_n(pred_tokens, ref_tokens, n=2)
        scores["rougeL"] += rouge_l(pred_tokens, ref_tokens)
    return {key: value / max(len(predictions), 1) for key, value in scores.items()}


def xlsum_metric_fn(p):
    """
    Token-level accuracy from teacher-forced logits, for the generic evaluation
    loop (proper ROUGE evaluation is done with generation in evaluate_xlsum)
    """
    if isinstance(p, dict):
        predictions = p["logits"]
        labels = p["labels"]
    else:
        predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    mask = labels != -100
    return {
        "token_accuracy": np.count_nonzero((predictions == labels) & mask)
        / max(np.count_nonzero(mask), 1)
    }


def xlsum_clm_metric_fn(p):
    """
    Same as xlsum_metric_fn but for decoder-only models, where logits at
    position i predict the label at position i+1
    """
    if isinstance(p, dict):
        predictions = p["logits"]
        labels = p["labels"]
    else:
        predictions, labels = p
    return xlsum_metric_fn((predictions[:, :-1], labels[:, 1:]))
