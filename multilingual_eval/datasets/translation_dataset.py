from typing import Dict, Union, List, Optional

from datasets import get_dataset_infos, load_dataset, get_dataset_config_names


def get_news_commentary(lang1: str, lang2: str, cache_dir=None):
    """
    Load and normalize the news_commentary dataset for a give pair of language
    """

    subsets = set(get_dataset_infos("news_commentary").keys())

    candidate_subset = "-".join(sorted([lang1, lang2]))

    if candidate_subset not in subsets:
        raise Exception(f"pair {candidate_subset} is not available in the `new_commentary` dataset")

    news_commentary = load_dataset(
        "news_commentary", candidate_subset, streaming=True, cache_dir=cache_dir
    )["train"]

    def preprocess_news_commentary(example):
        return {k: v for k, v in example["translation"].items()}

    news_commentary = (
        news_commentary.map(preprocess_news_commentary)
        .remove_columns(["id", "translation"])
        .filter(lambda x: x[lang1] is not None and x[lang2] is not None)
    )

    return news_commentary


def get_opus100(lang1: str, lang2: str, split="train", cache_dir=None):
    """
    Load and normalize the opus100 dataset for a give pair of language
    """

    subsets = set(get_dataset_config_names("opus100"))

    candidate_subset = "-".join(sorted([lang1, lang2]))

    if candidate_subset not in subsets:
        raise Exception(f"pair {candidate_subset} is not available in the `opus100` dataset")

    def preprocess_opus100(example):
        if "translation" in example:
            return {k: v for k, v in example["translation"].items()}
        return example

    opus100 = load_dataset("opus100", candidate_subset, streaming=True, cache_dir=cache_dir)[split]

    opus100 = (
        opus100.map(preprocess_opus100)
        .remove_columns(["translation"])
        .filter(lambda x: x.get(lang1) is not None and x.get(lang2) is not None)
    )

    return opus100


def get_translation_langs(dataset_name: str):
    """
    Get available language pairs for a given translation dataset
    """
    return list(map(lambda x: tuple(x.split("-")), get_dataset_config_names(dataset_name)))
