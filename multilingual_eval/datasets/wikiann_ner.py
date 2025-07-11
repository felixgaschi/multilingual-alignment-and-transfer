import logging
from typing import List, Union
import numpy as np

from datasets import load_dataset
from evaluate import load as load_metric


from multilingual_eval.datasets.token_classification import get_token_classification_getter
from multilingual_eval.datasets.lang_preprocessing import StanfordSegmenterWithLabelAlignmentMapper


class ReplaceByFileEntryMapper:
    """
    Callable for Dataset.map that replaces the HF-hosted wikiann entries with
    entries from a file. Useful for debugging
    """

    _str_labels = dict(zip(["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"], range(7)))

    def __init__(self, fname: str, label_name="ner_tags"):
        self.fname = fname
        self.iterator = None
        self.label_name = label_name

    def loop_over_file(self):
        with open(self.fname, "r") as f:
            tokens = []
            labels = []
            for line in f:
                if len(line.strip()) == 0:
                    yield {"tokens": tokens, self.label_name: labels}
                    tokens = []
                    labels = []
                    continue
                word, label = line.strip().split("\t")
                _, word = word.split(":", 1)
                tokens.append(word)
                labels.append(self._str_labels[label])
        if tokens:
            yield {"tokens": tokens, self.label_name: labels}

    def __enter__(self):
        self.iterator = self.loop_over_file()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.iterator = None

    def __call__(self, examples):
        new_examples = [next(self.iterator) for _ in range(len(examples["tokens"]))]

        return {
            "tokens": [elt["tokens"] for elt in new_examples],
            self.label_name: [elt[self.label_name] for elt in new_examples],
        }

    @classmethod
    def get_language_specific_dataset_transformer(
        cls, language: str, fname: str, label_name="ner_tags"
    ):
        def language_specific_transformer(lang, dataset):
            if lang == language:
                with cls(fname, label_name=label_name) as mapper:
                    return dataset.map(mapper, batched=True)
            return dataset

        return language_specific_transformer


get_wikiann_ner = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "wikiann",
        lang,
        cache_dir=cache_dir,
    ),
    "ner_tags",
)


def ner_post_processing(labels):
    """
    Post processing following Wu et al. for NER output
    to avoid predictions that are incompatible with BIO format
    """

    new_labels = []

    # Replace standalone I-X by B-X
    previous = None
    for i, label in enumerate(labels):
        if label[0] == "I" and (previous is None or previous == "O"):
            new_labels.append(f"B-{label[2:]}")
        else:
            new_labels.append(label)
        previous = label

    # Replace B-X I-Y I-Z by B-Z I-Z I-Z
    previous = None
    for i, label in zip(list(range(len(new_labels)))[::-1], new_labels[::-1]):
        if previous is None and label[0] == "I":
            previous = label[2:]
        elif label == "O":
            previous = None
        elif previous is not None and label[2:] != previous:
            new_labels[i] = f"{label[0]}-{previous}"
            if label[0] == "B":
                previous = None

    return new_labels


def get_wikiann_metric_fn():
    """
    Get dedicated metrics for the wikiann dataset
    """

    metric = load_metric("seqeval")

    str_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def compute_metrics(p):
        if isinstance(p, dict):
            predictions = p["logits"]
            labels = p["labels"]
        else:
            predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        raw_true_predictions = [
            [str_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_predictions = list(map(ner_post_processing, raw_true_predictions))

        true_labels = [
            [str_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        logging.debug(f"raw predictions: {raw_true_predictions[:5]}")
        logging.debug(f"post-processed predictions: {true_predictions[:5]}")
        logging.debug(f"labels: {true_labels[:5]}")

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics
