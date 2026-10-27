from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from datasets.iterable_dataset import IterableDataset, ExamplesIterable
from datasets import interleave_datasets, concatenate_datasets
from typing import List, Tuple, Optional, Dict
import os
from collections import defaultdict
import contextlib
import itertools
import numpy as np
import logging

from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset


def get_pharaoh_dataset(
    translation_file: str,
    alignment_file: Optional[str],
    repeat=False,
    keep_only_one_to_one=True,
    ignore_identical=True,
):
    def pharaoh_reader(offset=0):
        with open(translation_file) as translation_reader, (open(alignment_file) if alignment_file else contextlib.nullcontext()) as alignment_reader:
            if alignment_file is None:
                alignment_iterator = itertools.repeat(None)
            else:
                alignment_iterator = alignment_reader
            for i, (translation, alignment) in enumerate(zip(translation_reader, alignment_iterator)):
                parts = translation.split("|||")
                if len(parts) != 2:
                    continue
                left_sent, right_sent = parts
                left_tokens = left_sent.strip().split()
                right_tokens = right_sent.strip().split(" ")

                output = {
                    "left_tokens": left_tokens,
                    "right_tokens": right_tokens,
                }

                if alignment_file is not None:
                    pairs = alignment.strip().split()
                    pairs = list(map(lambda x: tuple(map(int, x.split("-"))), pairs))

                    left_positions = list(map(lambda x: x[0], pairs))
                    right_positions = list(map(lambda x: x[1], pairs))

                    if keep_only_one_to_one:
                        new_left_positions = []
                        new_right_positions = []
                        for a, b in zip(left_positions, right_positions):
                            if left_positions.count(a) > 1 or right_positions.count(b) > 1:
                                continue
                            new_left_positions.append(a)
                            new_right_positions.append(b)
                        left_positions = new_left_positions
                        right_positions = new_right_positions

                    if ignore_identical:
                        new_left_positions = []
                        new_right_positions = []
                        for a, b in zip(left_positions, right_positions):
                            if left_tokens[a] == right_tokens[b]:
                                continue
                            new_left_positions.append(a)
                            new_right_positions.append(b)
                        left_positions = new_left_positions
                        right_positions = new_right_positions

                    if len(left_positions) == 0:
                        continue

                    output.update({
                        "aligned_left_ids": left_positions,
                        "aligned_right_ids": right_positions,
                    })

                yield offset + i, output

        if repeat:
            yield from pharaoh_reader(offset=i + 1)

    return IterableDataset(ExamplesIterable(pharaoh_reader, {}))


def get_multiparallel_pharaoh_dataset(
    translation_files: List[str],
    alignment_files: List[str],
    repeat=False,
    keep_only_one_to_one=True,
    ignore_identical=False,
):
    def multiparallel_reader():
        with contextlib.ExitStack() as stack:
            translation_readers = [stack.enter_context(open(fname)) for fname in translation_files]
            alignment_readers = [stack.enter_context(open(fname)) for fname in alignment_files]

            for i, (translations, alignments) in enumerate(
                zip(zip(*translation_readers), zip(*alignment_readers))
            ):
                parts = list(map(lambda x: x.split("|||"), translations))
                if any(map(lambda x: len(x) != 2, parts)):
                    continue

                left_tokens = list(
                    map(lambda x: list(filter(lambda y: len(y) > 0, x[0].split())), parts)
                )
                right_tokens = list(
                    map(lambda x: list(filter(lambda y: len(y) > 0, x[1].split())), parts)
                )

                pairs = list(map(lambda x: x.strip().split(), alignments))
                pairs = list(
                    map(lambda y: list(map(lambda x: tuple(map(int, x.split("-"))), y)), pairs)
                )

                left_positions = list(map(lambda y: list(map(lambda x: x[0], y)), pairs))
                right_positions = list(map(lambda y: list(map(lambda x: x[1], y)), pairs))

                # We only keep pairs involving left words that are involved
                # in one and only one pair
                new_left_positions = []
                new_right_positions = []
                for single_left_pos, single_right_pos in zip(left_positions, right_positions):
                    new_single_left = []
                    new_single_right = []
                    for a, b in zip(single_left_pos, single_right_pos):
                        if single_left_pos.count(a) > 1:
                            continue
                        new_single_left.append(a)
                        new_single_right.append(b)
                    new_left_positions.append(new_single_left)
                    new_right_positions.append(new_single_right)

                left_positions = new_left_positions
                right_positions = new_right_positions

                if keep_only_one_to_one:
                    new_left_positions = []
                    new_right_positions = []
                    for single_left_pos, single_right_pos in zip(left_positions, right_positions):
                        new_single_left = []
                        new_single_right = []
                        for a, b in zip(single_left_pos, single_right_pos):
                            if single_right_pos.count(b) > 1:
                                continue
                            new_single_left.append(a)
                            new_single_right.append(b)
                        new_left_positions.append(new_single_left)
                        new_right_positions.append(new_single_right)

                    left_positions = new_left_positions
                    right_positions = new_right_positions

                if ignore_identical:
                    new_left_positions = []
                    new_right_positions = []
                    for i, (single_left_pos, single_right_pos) in enumerate(
                        zip(left_positions, right_positions)
                    ):
                        new_single_left = []
                        new_single_right = []
                        for a, b in zip(single_left_pos, single_right_pos):
                            if left_tokens[i][a] == right_tokens[i][b]:
                                continue
                            new_single_left.append(a)
                            new_single_right.append(b)
                        new_left_positions.append(new_single_left)
                        new_right_positions.append(new_single_right)

                    left_positions = new_left_positions
                    right_positions = new_right_positions

                # Keep only pairs involving left words involved in all languages
                left_positions_sets = list(map(set, left_positions))
                common_left_positions = left_positions_sets[0].intersection(
                    *left_positions_sets[1:]
                )

                left_positions = list(
                    map(
                        lambda y: list(
                            map(lambda x: x[0], filter(lambda x: x[0] in common_left_positions, y))
                        ),
                        pairs,
                    )
                )
                right_positions = list(
                    map(
                        lambda y: list(
                            map(lambda x: x[1], filter(lambda x: x[0] in common_left_positions, y))
                        ),
                        pairs,
                    )
                )

                if len(left_positions[0]) == 0:
                    continue

                yield i, {
                    "left_tokens": left_tokens[0],
                    "right_tokens": right_tokens,
                    "aligned_left_ids": left_positions[0],
                    "aligned_right_ids": right_positions,
                }
        if repeat:
            yield from multiparallel_reader()

    return IterableDataset(ExamplesIterable(multiparallel_reader, {}))


class AdaptAlignmentToTokenizerMapper:
    """
    Class for a dataset mapper that adapts word positions from the DatasetMapperForRealignment
    according to a model-specific tokenization, a bit like when realigning labels of a token-wise
    classification
    """

    def __init__(
        self,
        tokenizer,
        max_length=None,
        remove_underscore_if_roberta=True,
        first_subword_only=True,
        noaligner=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.first_subword_only = first_subword_only
        self.remove_underscore = remove_underscore_if_roberta and isinstance(
            self.tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)
        )
        if self.remove_underscore:
            self.underscore_id = self.tokenizer.convert_tokens_to_ids(["▁"])[0]
        self.noaligner = noaligner

    def __call__(self, examples):
        """
        Take batch of sentences with aligned words extracted from a translation dataset
        (left_tokens, right_tokens, aligned_left_ids, and aligned_right_ids) and return
        a sample for a realignment task, with properties:
        - left_*: (like left_input_ids) result of the tokenizer for the left sentence
        - right_*: same for the right sentence
        - alignment_left_positions: position range of all words in the left sentence (in term of subword)
        - alignment_right_positions: same for the right sentence
        - alignment_left_ids: index of aligned word in alignment_left_positions
        - alignment_right_ids: index of corresponding aligned words in alignment_right_positions
        - alignment_nb: the number of aligned pair (usefull for truncation)
        - alignment_left_length: the number of word in alignment_left_positions (usefull for truncation)
        - alignment_right_length: the same for the right sentence
        """
        left_tokens = examples["left_tokens"]
        right_tokens = examples["right_tokens"]
        
        left_tokenized_inputs = self.tokenizer(
            left_tokens, truncation=True, is_split_into_words=True, max_length=self.max_length
        )
        right_tokenized_inputs = self.tokenizer(
            right_tokens, truncation=True, is_split_into_words=True, max_length=self.max_length
        )

        alignment_left_pos, new_left_inputs = self.retrieve_positions_and_remove_underscore(
            left_tokenized_inputs, left_tokens
        )
        alignment_right_pos, new_right_inputs = self.retrieve_positions_and_remove_underscore(
            right_tokenized_inputs, right_tokens
        )

        if self.noaligner:
            alignment_left_pos = [
                [pos for pos in sent if pos is not None] for sent in alignment_left_pos
            ]
            alignment_right_pos = [
                [pos for pos in sent if pos is not None] for sent in alignment_right_pos
            ]

        output = {
            **{f"left_{k}": v for k, v in new_left_inputs.items()},
            **{f"right_{k}": v for k, v in new_right_inputs.items()},
            "alignment_left_positions": alignment_left_pos,
            "alignment_right_positions": alignment_right_pos,
            "alignment_left_length": [len(elt) for elt in alignment_left_pos],
            "alignment_right_length": [len(elt) for elt in alignment_right_pos],
        }

        if self.noaligner:
            return output

        aligned_left_ids = examples["aligned_left_ids"]
        aligned_right_ids = examples["aligned_right_ids"]

        (
            alignment_left_pos,
            alignment_right_pos,
            aligned_left_ids,
            aligned_right_ids,
        ) = self.realign_pairs(
            alignment_left_pos, alignment_right_pos, aligned_left_ids, aligned_right_ids
        )

        return {
            **output,
            "alignment_left_positions": alignment_left_pos,
            "alignment_right_positions": alignment_right_pos,
            "alignment_left_length": [len(elt) for elt in alignment_left_pos],
            "alignment_right_length": [len(elt) for elt in alignment_right_pos],
            "alignment_left_ids": aligned_left_ids,
            "alignment_right_ids": aligned_right_ids,
            "alignment_nb": [len(elt) for elt in aligned_left_ids],
        }

    def retrieve_positions_and_remove_underscore(self, tokenized_inputs, tokens):
        """
        Retrieve positions of words in term of subword and remove special Roberta underscore
        if needed
        """
        if self.remove_underscore:
            result = {key: [] for key in tokenized_inputs}
        else:
            result = tokenized_inputs

        n = len(tokenized_inputs["input_ids"])

        positions = [[None for _ in sent] for sent in tokens]
        for i in range(n):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            if self.remove_underscore:
                result_sample = {key: [] for key in tokenized_inputs if key in result}
            delta = 0
            for j, word_idx in enumerate(word_ids):
                if self.remove_underscore:
                    if tokenized_inputs["input_ids"][i][j] == self.underscore_id:
                        delta += -1
                        continue
                    for key in result_sample:
                        result_sample[key].append(tokenized_inputs[key][i][j])

                if word_idx is None:
                    continue

                if positions[i][word_idx] is None:
                    positions[i][word_idx] = [j + delta, j + delta + 1]
                elif not self.first_subword_only:
                    positions[i][word_idx][1] = j + delta + 1

            if self.remove_underscore:
                for key, value in result_sample.items():
                    result[key].append(value)

        return positions, result

    def realign_pairs(
        self, alignment_left_pos, alignment_right_pos, alignment_left_ids, alignment_right_ids
    ):
        """
        Remove null values in positions of words (in alignment_left_pos and alignment_right_pos)
        and modifies alignment_left_ids and alignment_right_ids accordingly
        """

        new_alignment_left_pos = []
        new_alignment_right_pos = []
        new_alignment_left_ids = []
        new_alignment_right_ids = []

        n = len(alignment_left_pos)
        for i in range(n):
            alignment_left_pos_entry = []
            alignment_right_pos_entry = []

            left_old_to_new_pos = []
            right_old_to_new_pos = []

            for pos in alignment_left_pos[i]:
                if pos is None:
                    left_old_to_new_pos.append(None)
                else:
                    alignment_left_pos_entry.append(pos)
                    left_old_to_new_pos.append(len(alignment_left_pos_entry) - 1)

            for pos in alignment_right_pos[i]:
                if pos is None:
                    right_old_to_new_pos.append(None)
                else:
                    alignment_right_pos_entry.append(pos)
                    right_old_to_new_pos.append(len(alignment_right_pos_entry) - 1)

            alignment_left_ids_entry = []
            alignment_right_ids_entry = []

            for left_idx, right_idx in zip(alignment_left_ids[i], alignment_right_ids[i]):
                if (
                    left_old_to_new_pos[left_idx] is not None
                    and right_old_to_new_pos[right_idx] is not None
                ):
                    alignment_left_ids_entry.append(left_old_to_new_pos[left_idx])
                    alignment_right_ids_entry.append(right_old_to_new_pos[right_idx])

            new_alignment_left_pos.append(alignment_left_pos_entry)
            new_alignment_right_pos.append(alignment_right_pos_entry)
            new_alignment_left_ids.append(alignment_left_ids_entry)
            new_alignment_right_ids.append(alignment_right_ids_entry)

        return (
            new_alignment_left_pos,
            new_alignment_right_pos,
            new_alignment_left_ids,
            new_alignment_right_ids,
        )


class MultiparallelAdaptAlignmentToTokenizerMapper(AdaptAlignmentToTokenizerMapper):
    def __call__(self, examples):
        # Not really optimal but simpler to implement
        results = []

        n_langs = len(examples["right_tokens"][0])
        batch_size = len(examples["right_tokens"])

        for i in range(n_langs):
            new_examples = {
                k: list(map(lambda x: x[i], v)) if k in ["right_tokens", "aligned_right_ids"] else v
                for k, v in examples.items()
            }
            results.append(super().__call__(new_examples))

        alignment_left_ids = []
        alignment_right_ids = []
        alignment_nb = []

        for i_batch in range(batch_size):
            left_ids = list(
                set(results[0]["alignment_left_ids"][i_batch]).intersection(
                    *[results[j]["alignment_left_ids"][i_batch] for j in range(1, n_langs)]
                )
            )
            id_position = {idx: i for i, idx in enumerate(left_ids)}
            right_ids = [[None] * len(id_position) for _ in range(n_langs)]
            for i_lang in range(n_langs):
                for left_id, right_id in zip(
                    results[i_lang]["alignment_left_ids"][i_batch],
                    results[i_lang]["alignment_right_ids"][i_batch],
                ):
                    if left_id in id_position:
                        right_ids[i_lang][id_position[left_id]] = right_id

            alignment_left_ids.append(left_ids)
            alignment_right_ids.append(right_ids)
            alignment_nb.append(len(left_ids))

        # Do not forget that it is batched !
        return {
            **{k: v for k, v in results[0].items() if k.startswith("left_")},
            **{
                k: list(map(list, zip(*map(lambda x: x[k], results))))
                for k in results[0].keys()
                if k.startswith("right_")
            },  # zip(*list) is like a transposition
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": results[0]["alignment_left_positions"],
            "alignment_right_positions": list(
                map(list, zip(*map(lambda x: x["alignment_right_positions"], results)))
            ),
            "alignment_nb": alignment_nb,
            "alignment_left_length": results[0]["alignment_left_length"],
            "alignment_right_length": list(
                map(list, zip(*map(lambda x: x["alignment_right_length"], results)))
            ),
        }


def get_realignment_dataset_for_one_pair(
    tokenizer,
    translation_file: str,
    alignment_file: Optional[str],
    max_length=None,
    first_subowrd_only=True,
    ignore_identical=True,
    seed=None,
    left_id=None,
    right_id=None,
):
    """
    Get realignment dataset from a parallel dataset in FastAlign format and FastAlign output
    """
    raw_dataset = get_pharaoh_dataset(
        translation_file, alignment_file, ignore_identical=ignore_identical, repeat=True
    ).shuffle(seed=seed, buffer_size=10_000)
    mapper = AdaptAlignmentToTokenizerMapper(
        tokenizer, max_length=max_length, first_subword_only=first_subowrd_only, noaligner=alignment_file is None
    )
    dataset = raw_dataset.map(
        mapper,
        batched=True,
        remove_columns=["aligned_left_ids", "aligned_right_ids", "left_tokens", "right_tokens"] if alignment_file is not None else ["left_tokens", "right_tokens"],
    ).filter(lambda x: x["alignment_left_length"] > 0 and x["alignment_right_length"] > 0 and (alignment_file is None or len(x["alignment_left_ids"]) > 0))
    if left_id is not None:
        dataset = dataset.map(lambda x: {**x, "left_lang_id": [left_id]})
    if right_id is not None:
        dataset = dataset.map(lambda x: {**x, "right_lang_id": [right_id]})
    return dataset.with_format("torch")


def get_multiparallel_realignment_dataset(
    tokenizer,
    translation_files: List[str],
    alignment_files: List[str],
    max_length=None,
    first_subowrd_only=True,
    ignore_identical=False,
    seed=None,
    left_id=None,
    right_id=None,
):
    raw_dataset = get_multiparallel_pharaoh_dataset(
        translation_files, alignment_files, ignore_identical=ignore_identical
    ).shuffle(seed=seed, buffer_size=10_000)
    mapper = MultiparallelAdaptAlignmentToTokenizerMapper(
        tokenizer, max_length=max_length, first_subword_only=first_subowrd_only
    )
    dataset = raw_dataset.map(
        mapper,
        batched=True,
        remove_columns=["aligned_left_ids", "aligned_right_ids", "left_tokens", "right_tokens"],
    ).filter(lambda x: len(x["alignment_left_ids"]) > 0)

    def get_map_fn(i):
        def the_map_fn(x):
            return {
                k: v[i]
                for k, v in x.items()
                if k.startswith("right_") or k.startswith("alignment_right_")
            }

        return the_map_fn

    datasets = [
        dataset.map(
            get_map_fn(i),
            batched=False,
        )
        for i in range(len(translation_files))
    ]

    if left_id is not None:
        datasets = [ds.map(lambda x: {**x, "left_lang_id": [left_id]}) for ds in datasets]
    if right_id is not None:
        datasets = [
            ds.map(lambda x: {**x, "right_lang_id": [idx]}) for ds, idx in zip(datasets, right_id)
        ]
    return [ds.with_format("torch") for ds in datasets]


def get_multilingual_realignment_dataset(
    tokenizer,
    translation_path: str,
    alignment_path: Optional[str],
    pairs: List[Tuple[str, str]],
    max_length=None,
    seed=None,
    lang_to_id: Optional[Dict[str, int]] = None,
    split="train",
    return_torch_compatible=True,
    do_interleave_datasets=True,
    first_subowrd_only=True,
    return_iterator_list=False # Return an iterator for each language, together as a list, to be used in weighted sampling
):
    """
    Load and prepare (lazily) a realignment dataset based on:
    - tokenizer
    - translation_path, the directory where parallel data in pharaoh format (source ||| target) can be found in files of the form "{left_lang}-{right_lang}.tokenized.{split}.txt"
    - alignment_path, the directory where corresponding alignment typically produced by FastAlign or AWESOME-align can be found in files of the form "{left_lang}-{right_lang}.{split}"
    - pairs: list of tuples of source-target language pairs
    - max_length
    - seed: seed for shuffling the dataset
    - lang_to_id: an optional dictionary to attribute ids to different languages (a default one will be computed otherwise)
    - split: (deprecated) specifies the split of the alignment dataset (cf. translation and alignment file formats)
    - return_torch_compatible: whether to convert the dataset into a pytorch-compatible one
    - do_interleave_datasets: True to interleave all datasets and False to return a list of datasets (one for each language pair)
    """
    lang_to_id = lang_to_id or defaultdict(lambda: None)
    datasets = [
        get_realignment_dataset_for_one_pair(
            tokenizer,
            os.path.join(translation_path, f"{left_lang}-{right_lang}.tokenized.{split}.txt"),
            os.path.join(alignment_path, f"{left_lang}-{right_lang}.{split}") if alignment_path else None,
            max_length=max_length,
            seed=seed,
            left_id=lang_to_id[left_lang],
            right_id=lang_to_id[right_lang],
            first_subowrd_only=first_subowrd_only,
        )
        for left_lang, right_lang in pairs
    ]
    if return_iterator_list:
        return [TorchCompatibleIterableDataset(one_pair_dataset) for one_pair_dataset in datasets]

    if not do_interleave_datasets:
        datasets = concatenate_datasets(datasets, axis=0)
        if return_torch_compatible:
            datasets = TorchCompatibleIterableDataset(datasets)
        return datasets

    dataset = interleave_datasets(datasets, seed=seed, probabilities=[1.0 / len(datasets)] * len(datasets))

    if return_torch_compatible:
        dataset = TorchCompatibleIterableDataset(dataset)

    return dataset
