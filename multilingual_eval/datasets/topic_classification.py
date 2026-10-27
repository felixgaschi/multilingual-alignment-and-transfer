from typing import Union, List
import numpy as np
from datasets import interleave_datasets
from huggingface_hub import snapshot_download

from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset, load_dataset_cached

SIB200_CATEGORIES = sorted(["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"])

class TopicClassificationMapper:
    def __init__(self, tokenizer, max_length=None, text_column="text", label_column="label", label_to_id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.label_to_id = label_to_id

    def __call__(self, examples):
        res = self.tokenizer(
            examples[self.text_column], 
            max_length=self.max_length, 
            truncation=True
        )
        
        if self.label_to_id is not None:
            res["label"] = [self.label_to_id[lbl] for lbl in examples[self.label_column]]
        else:
            res["label"] = examples[self.label_column]
            
        return res


def get_topic_classification(
    lang: Union[List[str], str],
    tokenizer,
    limit=None,
    ds_name="Davlan/sib200",
    split="train",
    datasets_cache_dir=None,
    interleave=True,
    lang_id=None,
    return_length=False,
    n_epochs=1,
    remove_useless=True,
    max_length=256,
    text_column="text",
    label_column="category",
):
    """
    Return Topic Classification dataset
    """
    if lang == "en": lang = "eng_Latn"

    if not isinstance(lang, list):
        lang = [lang]
    if lang_id is not None:
        if not isinstance(lang_id, list):
            lang_id = [lang_id]
        assert len(lang_id) == len(lang)

    datasets = [load_dataset_cached(ds_name, elt, cache_dir=datasets_cache_dir)[split] for elt in lang]

    n_datasets = len(datasets)

    if limit:
        limits = [
            limit // n_datasets + (1 if i < limit % n_datasets else 0) for i in range(n_datasets)
        ]

        datasets = map(
            lambda x: x[0].shuffle().filter(lambda _, i: i < x[1], with_indices=True),
            zip(datasets, limits),
        )

    if n_datasets == 1:
        datasets = [next(iter(datasets))]
    elif interleave:
        datasets = [interleave_datasets(datasets)]

    if return_length:
        lengths = list(map(len, datasets))

    if n_epochs > 1:
        datasets = map(lambda x: convert_dataset_to_iterable_dataset(x, n_epochs), datasets)

    # Inspect label type
    sample_label = datasets[0][0][label_column]
    label_to_id = None

    if not isinstance(sample_label, (int, np.integer)):
        unique_categories = set()
        if ds_name == "Davlan/sib200":
            unique_categories = SIB200_CATEGORIES
        else:
            for ds in datasets:
                unique_categories.update(ds.unique(label_column))
        
        sorted_categories = sorted(list(unique_categories))
        label_to_id = {cat: i for i, cat in enumerate(sorted_categories)}

    datasets = list(
        map(
            lambda x: x.map(
                TopicClassificationMapper(
                    tokenizer,
                    max_length=max_length,
                    text_column=text_column,
                    label_column=label_column,
                    label_to_id=label_to_id,
                ),
                batched=True,
                remove_columns=x.column_names if remove_useless else None,
            ),
            datasets,
        ),
    )

    if lang_id is not None:
        datasets = list(
            map(lambda x: x[0].map(lambda y: {**y, "lang_id": [x[1]]}), zip(datasets, lang_id))
        )

    # if remove_useless:
    #     datasets = list(
    #         map(
    #             lambda x: x.remove_columns([text_column]),
    #             datasets,
    #         )
    #     )

    if n_datasets == 1 or interleave:
        if return_length:
            return datasets[0], lengths[0]
        return datasets[0]
    if return_length:
        return datasets, lengths
    return datasets