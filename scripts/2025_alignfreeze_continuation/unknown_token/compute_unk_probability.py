import os, sys

from transformers import AutoTokenizer
from contextlib import ExitStack

sys.path.append(os.curdir)

from multilingual_eval.datasets.dispatch_datasets import (
    get_dataset_fn,
)
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter

models = [
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base"
]

def get_unk_count(tokenizer, lang: str, dataset_name: str, split="test", zh_segmenter=None, data_cache_dir=None):

    
        
    dataset = get_dataset_fn(dataset_name, zh_segmenter=zh_segmenter)(
        lang,
        tokenizer,
        split=split,
        datasets_cache_dir=data_cache_dir
    )

    n_unk = 0
    n_total = 0
    for elt in dataset:
        input_ids = elt["input_ids"][1:-1]
        n_unk += input_ids.count(tokenizer.unk_token_id)
        n_total += len(input_ids)
    
    return n_unk / n_total


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--langs", type=str, nargs="+", default=None)
    parser.add_argument("--tokenizers", type=str, nargs="+", default=models)
    args = parser.parse_args()

    langs = args.langs or (
        ["bg", "cs", "de", "es", "lv", "af", "ar", "ca", "da", "el", "fa", "fi", "fr", "he", "hi", "hu", "it", "ja", "ko", "lt", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "ta", "th", "tr", "uk", "vi", "zh"]
        if args.dataset_name == "udpos"
        else
        ["ar", "bg", "de", "el", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh"]
    )

    with ExitStack() as stack:
        if "zh" in langs:
            zh_segmenter = stack.enter_context(StanfordSegmenter())
        else:
            zh_segmenter = None

    tokenizers = [AutoTokenizer.from_pretrained(name) for name in args.tokenizers]

    results = {}
    
    for tokenizer in tokenizers:
        for lang in langs:
            ratio = get_unk_count(tokenizer, lang, args.dataset_name, zh_segmenter=zh_segmenter)
            results[(tokenizer.name_or_path, lang)] = ratio

    for key, value in results.items():
        print(f"{key}: {value}")
