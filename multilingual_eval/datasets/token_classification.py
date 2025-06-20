from typing import Union, List, Optional
import numpy as np
import pycountry

from datasets import interleave_datasets
from huggingface_hub import snapshot_download
from datasets import load_dataset
from multilingual_eval.datasets.code_switching import (
    get_dataset_with_code_swicthing,
)
from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset

from multilingual_eval.datasets.label_alignment import LabelAlignmentMapper
from multilingual_eval.datasets.lang_preprocessing import StanfordSegmenterWithLabelAlignmentMapper
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter


def get_token_classification_getter(
    subset_loader,
    label_name: str,
):
    """
    Return a function that would load a token classification dataset and perform the
    necessary transformation

    Arguments:
    - subset_loader: a function that takes two arguments 'lang' (positional) and 'cache_dir' (keyword) that will
        load the dataset for a given language (lang) using the provided cache directory (cache_dir) which is None by default
    - label_name: the name of the properties containing the labels as integers
    """

    def get_token_classification_dataset(
        lang: Union[List[str], str],
        tokenizer,
        limit=None,
        split="train",
        datasets_cache_dir=None,
        interleave=True,
        first_subword_only=True,
        lang_id=None,
        dictionaries_for_code_switching=None,
        return_length=False,
        n_epochs=1,
        max_length=128,
        return_overflowing_tokens=True,
        zh_segmenter: Optional[StanfordSegmenter] = None,
        resegment_zh=False,
    ):
        """
        Load a task classification dataset for a given lang

        Arguments:

        - lang: the language, can be a list of labels if we want to load several subsets
        - tokenizer
        - limit: a limit on the total number of samples, default to None (no limit)
        - split: the split (typically "train" or "validation"), default "train"
        - datasets_cache_dir: the cache directory for the load_dataset function
        - interleave: if several languages are provided, decides whether to interleave the different
            datasets or return them as element of a list (default to True)
        - first_subword_only: whether to perform classification on the first subword of each token
            or on each subword (default to True)
        - zh_segmenter: segmenter object use to resegment annotated text in chinses, when it is character tokenized
            only used if resegment_zh is True
        - resegment_zh: default False, resegment chinese annotated data and realign labels when it is
            character-tokenized. If True, zh_segmenter must be set
        """

        if resegment_zh and zh_segmenter is None:
            raise Exception(
                f"resegment_sh is True, so zh_segmenter must be set, whereas it is not."
            )

        if not isinstance(lang, list):
            lang = [lang]
        if lang_id is not None:
            if not isinstance(lang_id, list):
                lang_id = [lang_id]
            assert len(lang_id) == len(lang)

        if dictionaries_for_code_switching and not isinstance(
            dictionaries_for_code_switching[0], list
        ):
            dictionaries_for_code_switching = [dictionaries_for_code_switching]

        afri_langs = {
            'bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul'
        }

        masakhapos_lang = [elt for elt in lang if elt in afri_langs]
        lang = [elt for elt in lang if elt not in afri_langs]
        
        datasets = [subset_loader(elt, cache_dir=datasets_cache_dir)[split] for elt in lang]

        if "pos" in label_name and masakhapos_lang:
            print(f"Loading dataset from {split} split for {masakhapos_lang}")
            masakhapos_datasets = load_masakhapos(split, masakhapos_lang, datasets_cache_dir)
            datasets.extend(masakhapos_datasets)

        n_datasets = len(datasets)

        if limit:
            limits = [
                limit // n_datasets + (1 if i < limit % n_datasets else 0)
                for i in range(n_datasets)
            ]

            datasets = map(
                lambda x: x[0].shuffle().filter(lambda _, i: i < x[1], with_indices=True),
                zip(datasets, limits),
            )

        if resegment_zh:
            datasets = map(
                lambda x: x[1]
                if x[0] != "zh"
                else x[1].map(
                    StanfordSegmenterWithLabelAlignmentMapper(zh_segmenter, label_name=label_name)
                ),
                zip(lang, datasets),
            )

        if n_datasets == 1:
            datasets = [next(iter(datasets))]
        elif interleave:
            datasets = [interleave_datasets(datasets)]

        if return_length:
            lengths = list(map(len, datasets))

        if n_epochs > 1:
            datasets = map(lambda x: convert_dataset_to_iterable_dataset(x, n_epochs), datasets)

        if dictionaries_for_code_switching:
            datasets = map(
                lambda x: get_dataset_with_code_swicthing(
                    x[1], dictionaries_for_code_switching[x[0]]
                ),
                enumerate(datasets),
            )

        datasets = list(
            map(
                lambda x: x.map(
                    LabelAlignmentMapper(
                        tokenizer,
                        label_name=label_name,
                        first_subword_only=first_subword_only,
                        max_length=max_length,
                        return_overflowing_tokens=return_overflowing_tokens,
                    ),
                    batched=True,
                    remove_columns=x.column_names,
                ),
                datasets,
            ),
        )

        if return_overflowing_tokens:
            datasets = list(map(lambda x: x.remove_columns("overflow_to_sample_mapping"), datasets))

        if lang_id is not None:
            datasets = list(
                map(lambda x: x[0].map(lambda y: {**y, "lang_id": [x[1]]}), zip(datasets, lang_id))
            )

        if n_datasets == 1 or interleave:
            if return_length:
                return datasets[0], lengths[0]
            return datasets[0]
        if return_length:
            return datasets, lengths
        return datasets

    return get_token_classification_dataset

def load_masakhapos(split, lang, datasets_cache_dir):
    """
    Return masakhapos dataset from Masakhane (https://huggingface.co/datasets/masakhane/masakhapos)
    """
    # datasets version < 2.15 are unable to load this dataset directly
    local_dir  = snapshot_download(
        repo_id="masakhane/masakhapos", 
        repo_type="dataset", 
        local_dir=f"{datasets_cache_dir}/masakhapos",
    )
    print(f"Dataset loading script masakhane/masakhapos downloaded to: {local_dir}. Loading datasets...")
    datasets = [load_dataset(f"{local_dir}/masakhapos.py", name=elt, split=split, cache_dir=local_dir) for elt in lang] 
    return datasets

def get_token_classification_metrics():
    """
    Default token classification metric: accuracy
    """

    def compute_metrics(p):
        if isinstance(p, dict):
            predictions = p["logits"]
            labels = p["labels"]
        else:
            predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            p
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        true_labels = [
            l
            for prediction, label in zip(predictions, labels)
            for (p, l) in zip(prediction, label)
            if l != -100
        ]

        return {
            "accuracy": np.mean(
                list(map(lambda x: x[0] == x[1], zip(true_labels, true_predictions)))
            )
        }

    return compute_metrics
