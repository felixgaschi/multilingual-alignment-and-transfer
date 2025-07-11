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
    america_langs = {"aym", "bzd", "cni", "gn", "hch", "nah", "oto", "quy", "shp", "tar"}

    afrixnli_lang = [elt for elt in lang if elt in afri_langs]
    americasnli_lang = [elt for elt in lang if elt in america_langs]
    lang = [elt for elt in lang if elt not in afrixnli_lang and elt not in americasnli_lang]

    datasets = [load_dataset("xnli", elt, data_dir=datasets_cache_dir)[split] for elt in lang if elt != "ind" and elt != "mya"]

    if afrixnli_lang:
        print(f"Loading dataset from {split} split for {afrixnli_lang}")
        afrixnli_datasets = load_afrixnli(split, afrixnli_lang, datasets_cache_dir)
        datasets.extend(afrixnli_datasets)
    
    if americasnli_lang:
        americas_split = split
        if split not in ['validation', 'test']:
            logging.warning(f"Split {split} is not available for AmericasNLI. Defaulting to validation.")
            americas_split = 'validation'
        print(f"Loading dataset from {split} split for {americasnli_lang}")
        americasnli_datasets = [load_dataset("nala-cub/americas_nli", elt, cache_dir=datasets_cache_dir)[americas_split] for elt in americasnli_lang]
        datasets.extend(americasnli_datasets)

    if "ind" in lang:
        # The data is split across train, valid, test_lay, and test_expert. 
        # test_expert is written by expert annotators, whereas the rest are written by lay annotators.
        indo_split = "test_expert" if split == "test" else split
        datasets.append(load_dataset("afaji/indonli", data_dir=datasets_cache_dir, trust_remote_code=True)[indo_split])

    if "mya" in lang:
        # Load and preprocess Myanmar XNLI
        label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }

        mya_ds = (
            load_dataset("akhtet/myanmar-xnli", cache_dir=datasets_cache_dir)[split]
            .rename_columns({
                'sentence1_my': 'premise',
                'sentence2_my': 'hypothesis',
            })
            .map(lambda x: {"label": label_map[x["label"]]})
            .remove_columns(["sentence1_en", "sentence2_en", "genre"])
        )
        datasets.append(mya_ds)

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
        print(f"Split {split} is not supported for AfriXNLI. Defaulting to 'validation'")
        split = "validation"

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
