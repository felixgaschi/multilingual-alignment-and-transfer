from typing import Union, List
import numpy as np
from datasets import load_dataset, interleave_datasets
from huggingface_hub import snapshot_download

from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset


class XNLIMapper:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        res = self.tokenizer(
            examples["premise"], examples["hypothesis"], max_length=self.max_length, truncation=True
        )
        return {**res, "label": examples["label"]}


def get_xnli(
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
    max_length=256,
):
    """
    Return XNLI dataset
    """
    if not isinstance(lang, list):
        lang = [lang]
    if lang_id is not None:
        if not isinstance(lang_id, list):
            lang_id = [lang_id]
        assert len(lang_id) == len(lang)

    afri_langs = {'amh', 'eng', 'ewe', 'fra', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul'}

    afrixnli_lang = [elt for elt in lang if elt in afri_langs]
    lang = [elt for elt in lang if elt not in afri_langs]

    datasets = [load_dataset("xnli", elt, data_dir=datasets_cache_dir)[split] for elt in lang]

    if afri_langs:
        afrixnli_datasets = load_afrixnli(split, afrixnli_lang, datasets_cache_dir)
        datasets.extend(afrixnli_datasets)

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

    datasets = list(
        map(
            lambda x: x.map(
                XNLIMapper(
                    tokenizer,
                    max_length=max_length,
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
                lambda x: x.remove_columns(["premise", "hypothesis"]),
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

def load_afrixnli(split, lang, datasets_cache_dir):
    """
    Return AfriXNLI dataset from Masakhane (https://huggingface.co/datasets/masakhane/afrixnli)
    """

    def load_language_data(root_dir, lang):
        return load_dataset(
            "parquet",
            data_files={
                "test": f"{root_dir}/{lang}/test/*.parquet",
                "validation": f"{root_dir}/{lang}/validation/*.parquet"
            }
        )

    if split not in ["validation", "test"]:
        split = "validation"
        print(f"Split {split} is not supported for AfriXNLI. Defaulting to 'validation'")

    # datasets version < 2.15 are unable to load this dataset directly
    local_dir  = snapshot_download(
        repo_id="masakhane/afrixnli",
        repo_type="dataset",
        revision="refs/convert/parquet",
        local_dir=f"{datasets_cache_dir}/afrixnli",
    )
    print(f"Dataset masakhane/afrixnli downloaded to: {local_dir}")
    datasets = [load_language_data(f"{datasets_cache_dir}/afrixnli", elt)[split] for elt in lang]
    return datasets

def xnli_metric_fn(p):
    if isinstance(p, dict):
        predictions = p["logits"]
        labels = p["labels"]
    else:
        predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    return {"accuracy": np.count_nonzero(predictions == labels) / predictions.shape[0]}
