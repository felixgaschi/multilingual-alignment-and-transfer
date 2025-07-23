from typing import Union, List
from datasets import interleave_datasets
import evaluate
import logging

from huggingface_hub import snapshot_download

from multilingual_eval.datasets.data_utils import convert_dataset_to_iterable_dataset
from multilingual_eval.datasets.span_alignment import SpanAligner


def get_question_answering_getter(
    subset_loader,
):
    """
    Return a function that would load a question answering dataset and perform
    the required pre-processing

    Arguments:
    - subset_loader: a function that takes two arguments 'lang' (positional) and 'cache_dir' (keyword) that will
        load the dataset for a given language (lang) using the provided cache directory (cache_dir) which is None by default
    """

    def get_question_answering_dataset(
        lang: Union[List[str], str],
        tokenizer,
        limit=None,
        datasets_cache_dir=None,
        interleave=True,
        lang_id=None,
        split=None,
        return_length=False,
        n_epochs=1,
        max_length=128,
        stride=32,
        preprocessing=True,
    ):
        """
        Load a question answering dataset for a given lang

        Arguments:

        - lang: the language, can be a list of labels if we want to load several subsets
        - tokenizer
        - limit: a limit on the total number of samples, default to None (no limit)
        - datasets_cache_dir: the cache directory for the load_dataset function
        - interleave: if several languages are provided, decides whether to interleave the different
            datasets or return them as element of a list (default to True)
        """

        if not isinstance(lang, list):
            lang = [lang]
        if lang_id is not None:
            if not isinstance(lang_id, list):
                lang_id = [lang_id]
            assert len(lang_id) == len(lang)

        datasets = [subset_loader(elt, cache_dir=datasets_cache_dir) for elt in lang if elt != "ind"]

        if split is not None:
            datasets = list(map(lambda x: x[split], datasets))
        
        if "ind" in lang:
            indoqa_dataset = load_indoqa(split, datasets_cache_dir)
            datasets.extend(indoqa_dataset)

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

        if n_datasets == 1:
            datasets = [next(iter(datasets))]
        elif interleave:
            datasets = [interleave_datasets(datasets)]

        if return_length:
            lengths = list(map(len, datasets))

        if n_epochs > 1:
            datasets = map(lambda x: convert_dataset_to_iterable_dataset(x, n_epochs), datasets)

        if preprocessing:
            datasets = list(
                map(
                    lambda x: x.map(
                        SpanAligner(tokenizer, max_length=max_length, stride=stride),
                        batched=True,
                        remove_columns=x.column_names,
                    ),
                    datasets,
                )
            )

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
        print(len(datasets), split)
        print(datasets)
        return datasets

    return get_question_answering_dataset

def load_indoqa(split, datasets_cache_dir):
    """
    Return indoqa dataset from HuggingFace https://huggingface.co/datasets/jakartaresearch/indoqa
    """
    def transform_to_answers(example):
        return {
            "answers": {
                "text": [example["answer"]],
                "answer_start": [example["span_start"]],
                "answer_end": [example["span_end"]]
            }
        }
    from datasets import load_dataset
    
    if split not in ['train', 'validation']:
        logging.warning(f"Split {split} is not available. Defaulting to validation")
        split = "validation"
    
    # datasets version < 2.15 are unable to load this dataset directly
    local_dir = f"{datasets_cache_dir}/indoqa"
    local_dir  = snapshot_download(
            repo_id=f"jakartaresearch/indoqa", 
            repo_type="dataset", 
            local_dir=local_dir,
        )
    
    print(f"Dataset loading script jakartaresearch/indoqa downloaded to: {local_dir}. Loading datasets...")
    datasets = [
        load_dataset(f"{local_dir}/indoqa.py", split=split, cache_dir=local_dir, trust_remote_code=True)
        .filter(lambda example: example["category"] == "SPAN")
    ]
    datasets = [ds.map(transform_to_answers) for ds in datasets]
    return datasets


def get_question_answering_metrics():
    metric = evaluate.load("squad")

    def compute_metric(p):
        return metric.compute(predictions=p["logits"], references=p["label_ids"])

    return compute_metric
