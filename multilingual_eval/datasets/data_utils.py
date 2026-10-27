from datasets.iterable_dataset import IterableDataset
from datasets import load_dataset, load_from_disk
import torch
import itertools
import inspect
import hashlib
import json
import os
import functools

@functools.wraps(load_dataset)
def load_dataset_cached(*args, cache_dir: str | None = None, **kwargs):
    """
    Wrapper around load_dataset that saves the result to disk (under cache_dir)
    and loads from disk on subsequent calls, avoiding any network check.

    The on-disk directory is determined by a hash of the positional and keyword
    arguments forwarded to load_dataset, so different calls map to different dirs.
    cache_dir is NOT forwarded to load_dataset (pass it explicitly via kwargs if
    you also want the HF arrow cache to live there).
    """
    if cache_dir is None:
        return load_dataset(*args, **kwargs)

    key = json.dumps({"args": list(args), "kwargs": kwargs}, sort_keys=True)
    dir_hash = hashlib.md5(key.encode()).hexdigest()
    save_path = os.path.join(cache_dir, "saved_datasets", dir_hash)

    if os.path.isdir(save_path):
        return load_from_disk(save_path)

    dataset = load_dataset(*args, **kwargs)
    dataset.save_to_disk(save_path)
    return dataset

from multilingual_eval.seeds import seeds


def convert_dataset_to_iterable_dataset(dataset, repeat=1):
    """
    Convert a dataset.Dataset to an iterable dataset, giving the
    possibility to repeat the dataset several times with a different shuffle
    """

    return IterableDataset(
        enumerate(
            itertools.chain(
                *[
                    dataset
                    if i == 0
                    else dataset.shuffle(seed=seeds[i % len(seeds)] + i // len(seeds))
                    for i in range(repeat)
                ]
            )
        )
    )


def repeat_iterable_dataset(dataset, repeat):
    """
    Repeats an iterable dataset
    """

    return convert_dataset_to_iterable_dataset(dataset, repeat)


def infinite_iterable_dataset(dataset):
    """
    Provide an iterable for repeating an iterable dataset without exhaustion
    """

    return IterableDataset(enumerate(itertools.cycle(dataset)))


def get_signature_columns_if_needed(model, label_names=None):
    """
    Returns the name of the arguments expected by the model.forward method, useful
    to filter the column of a dataset before training
    taken from HF trainer:
    https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/trainer.py#L705
    """

    label_names = label_names or []
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    signature_columns += ["label", "label_ids", "labels", *label_names]
    return signature_columns


class TorchCompatibleIterableDataset(torch.utils.data.IterableDataset):
    """
    Converts a datasets.IterableDataset to torch.utils.data.IterableDataset
    for better compatibility with HF Trainer (oddly enough it does not work well
    with HF IterableDataset)
    """

    def __init__(self, normal_dataset):
        self.normal_dataset = normal_dataset

    def __iter__(self):
        return iter(self.normal_dataset)
