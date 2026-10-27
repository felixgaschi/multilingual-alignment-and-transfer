# Chinese load_dataset("un_multi", "en-zh", cache_dir=cache_dir)
# Others news_commentary
from collections import defaultdict
import itertools
import logging
from transformers import DataCollatorWithPadding, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List
import torch
from datasets import interleave_datasets
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader
import numpy as np

from multilingual_eval.data import get_dicos
from multilingual_eval.datasets.data_utils import (
    TorchCompatibleIterableDataset,
    convert_dataset_to_iterable_dataset,
    get_signature_columns_if_needed,
    infinite_iterable_dataset,
    repeat_iterable_dataset,
)
from multilingual_eval.datasets.fastalign_realignment import (
    get_fastalign_realignment_dataset_from_path,
)
from multilingual_eval.datasets.translation_dataset import get_news_commentary, get_opus100
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter
from multilingual_eval.datasets.xtreme_udpos import get_xtreme_udpos
from multilingual_eval.utils import (
    get_tokenizer_type,
    subwordlist_to_wordlist,
    LanguageSpecificTokenizer,
)


@dataclass
class BilingualDictionary:
    """
    Class for holding pairs of words in a bilingual dictionary
    """

    forward: Dict[Tuple[str], Set[Tuple[str]]]
    backward: Dict[Tuple[str], Set[Tuple[str]]]

    def sample_dictionary(self, fraction):
        """
        sample a dictionary, usefull for train/test splits
        """
        new_forward = defaultdict(lambda: set())
        new_backward = defaultdict(lambda: set())

        remaining_forward = defaultdict(lambda: set())
        remaining_backward = defaultdict(lambda: set())

        for key, values in self.forward.items():
            for value in values:
                if np.random.random() < fraction:
                    new_forward[key].add(value)
                    new_backward[value].add(key)
                else:
                    remaining_forward[key].add(value)
                    remaining_backward[value].add(key)

        self.forward = new_forward
        self.backward = new_backward

        return BilingualDictionary(remaining_forward, remaining_backward)


class DatasetMapperForRealignment:
    """
    Class for defining a callable that can be used as an argument of datasets.Dataset.map()
    Applied to a translation dataset, it will create an alignment table between tokens of
    the translated sentences
    """

    def __init__(
        self,
        left_key: str,
        right_key: str,
        dico_path=None,
        dico=None,
        dictionary_fraction=1.0,
        split="all",
        ignore_identical=True,
        add_identical=False,
        zh_segmenter: Optional[StanfordSegmenter] = None,
    ):
        if left_key == "zh" or right_key == "zh" and zh_segmenter is None:
            logging.warning(
                f"DatasetMapperForRealignment: segmenter is None whereas zh is one of the two languages passed. Will default to RegexTokenizer"
            )
        self.left_tokenizer = LanguageSpecificTokenizer(
            lang=left_key,
            zh_segmenter=zh_segmenter
        )
        self.right_tokenizer = LanguageSpecificTokenizer(
            lang=right_key,
            zh_segmenter=zh_segmenter
        )

        if dico is None:
            forward, backward = get_dicos(
                left_key,
                right_key,
                dico_path,
                ignore_identical=ignore_identical,
                tokenize=False,
                split=split,
            )

            self.dico = BilingualDictionary(forward, backward)
        else:
            self.dico = dico

        if dictionary_fraction < 1.0:
            self.other_dico = self.dico.sample_dictionary(dictionary_fraction)
            forward = self.dico.forward
            backward = self.dico.backward
        else:
            self.other_dico = self.dico

        self.left_key = left_key
        self.right_key = right_key

        self.add_identical = add_identical

    def __call__(self, example):
        """
        Take an translation sample of the form {"name_of_lang_1": "sentence", "name_of_lang_2": "sentence"}
        and return a sample for a realignment task, with properties:
        - left_tokens: list of tokens for the left language
        - right_tokens: same for right lang
        - aligned_left_ids: positions in left_tokens of aligned pairs
        - aligned_right_ids: positions in right_tokens of aligned pairs
        """
        left_sent = example[self.left_key]
        right_sent = example[self.right_key]

        left_tokens = self.left_tokenizer.tokenize(left_sent)
        right_tokens = self.right_tokenizer.tokenize(right_sent)

        aligned_left_ids, aligned_right_ids = self.find_aligned_words(left_tokens, right_tokens)

        if self.add_identical:
            (new_aligned_left_ids, new_aligned_right_ids,) = self.add_identical_words(
                left_tokens, right_tokens, aligned_left_ids, aligned_right_ids
            )
            aligned_left_ids += new_aligned_left_ids
            aligned_right_ids += new_aligned_right_ids

        return {
            "left_tokens": left_tokens,
            "right_tokens": right_tokens,
            "aligned_left_ids": aligned_left_ids,
            "aligned_right_ids": aligned_right_ids,
        }

    def find_aligned_words(self, left_tokens, right_tokens):
        """
        Compare tokens from both sentence and add them to the alignment table
        if they are in the bilingual dictionary (and if there is no ambiguity)
        Returns:
        - aligned_left_multi_ids: a list of int of size N indicating the position
            (in term of tokens) of each aligned element
        - aligned_right_multi_ids: potision of the translation of each word referenced by aligned_left_multi_pos
        """
        aligned_left_ids = []
        aligned_right_ids = []

        for left_pos, word in enumerate(left_tokens):
            candidates = self.dico.forward.get(word, set()).intersection(right_tokens)

            # Verify that there is one and only one candidate in set of words
            if len(candidates) != 1:
                continue

            # Verify that there is only one occurence of this word and extract it
            for i, (right_pos, right_word) in enumerate(
                filter(lambda x: x[1] in candidates, enumerate(right_tokens))
            ):
                if i == 1:
                    break
            if i != 0:
                continue

            # Verify that the target word is not the translation of another word in the source
            backward_candidates = self.dico.backward.get(right_word, set())
            counter = 0
            for w in left_tokens:
                if w in backward_candidates:
                    counter += 1
                if counter > 1:
                    break
            if counter != 1:
                continue

            aligned_left_ids.append(left_pos)
            aligned_right_ids.append(right_pos)
        return aligned_left_ids, aligned_right_ids

    def add_identical_words(self, left_tokens, right_tokens, aligned_left_ids, aligned_right_ids):
        """
        Add identical words found in both sentences
        """
        new_aligned_left_ids = []
        new_aligned_right_ids = []

        for left_pos, left_token in enumerate(left_tokens):
            if left_tokens.count(left_token) > 1:
                continue
            for right_pos, right_token in enumerate(right_tokens):
                if (
                    right_tokens.count(right_token) == 1
                    and left_token == right_token
                    and left_pos not in aligned_left_ids
                    and right_pos not in aligned_right_ids
                ):
                    new_aligned_left_ids.append(left_pos)
                    new_aligned_right_ids.append(right_pos)

        return new_aligned_left_ids, new_aligned_right_ids


class AdaptAlignmentToTokenizerMapper:
    """
    Class for a dataset mapper that adapts word positions from the DatasetMapperForRealignment
    according to a model-specific tokenization, a bit like when realigning labels of a token-wise
    classification
    """

    def __init__(
        self, tokenizer, max_length=None, remove_underscore_if_roberta=True, first_subword_only=True, noaligner=False
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


class DatasetMapperForInjectingRealignmentData:
    """
    deprecated: not useful with the new training loop
    Class for defining a callable that can be used as an argument of datasets.Dataset.map()
    Inject a realignment example inside the sample of a given task
    """

    def __init__(self, realignment_dataset):
        self.realignment_dataset = realignment_dataset
        self.realignment_iterator = iter(self.realignment_dataset)

    def __call__(self, example):
        try:
            realignment_example = next(self.realignment_iterator)
        except StopIteration:
            self.realignment_iterator = iter(self.realignment_dataset)
            realignment_example = next(self.realignment_iterator)

        return {**example, **realignment_example}


class RealignmentCollator:
    """
    Data collator for building and padding batch for the realignment task
    """

    def __init__(self, tokenizer, noaligner=False, **kwargs):
        self.usual_collator = DataCollatorWithPadding(tokenizer, **kwargs)
        self.noaligner = noaligner

    def __call__(self, examples):
        if isinstance(examples[0], tuple) and len(examples[0]) == 2:
            examples, lang_ids = zip(*examples)
        else:
            examples = examples
            lang_ids = None

        left_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("left_")}
            for sample in examples
        ]
        right_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("right_")}
            for sample in examples
        ]
        batch_left = {f"left_{k}": v for k, v in self.usual_collator(left_inputs).items()}
        batch_right = {f"right_{k}": v for k, v in self.usual_collator(right_inputs).items()}

        max_left_length = max(map(lambda x: x["alignment_left_length"], examples))
        max_right_length = max(map(lambda x: x["alignment_right_length"], examples))
        alignment_left_positions = torch.zeros(
            (len(examples), max_left_length, 2), dtype=torch.long
        )
        alignment_right_positions = torch.zeros(
            (len(examples), max_right_length, 2), dtype=torch.long
        )
        for i, ex in enumerate(examples):
            alignment_left_positions[i, : ex["alignment_left_length"]] = torch.LongTensor(
                ex["alignment_left_positions"]
            )
            alignment_right_positions[i, : ex["alignment_right_length"]] = torch.LongTensor(
                ex["alignment_right_positions"]
            )

        output = {
            **batch_left,
            **batch_right,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
            "alignment_left_length": torch.LongTensor(
                [ex["alignment_left_length"] for ex in examples]
            ),
            "alignment_right_length": torch.LongTensor(
                [ex["alignment_right_length"] for ex in examples]
            ),
        }
        if lang_ids is not None:
            output["lang_ids"] = torch.tensor(lang_ids)

        if self.noaligner:
            return output

        max_nb = max(map(lambda x: x["alignment_nb"], examples))
        alignment_left_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)
        alignment_right_ids = torch.zeros((len(examples), max_nb), dtype=torch.long)

        for i, ex in enumerate(examples):
            alignment_left_ids[i, : ex["alignment_nb"]] = torch.LongTensor(ex["alignment_left_ids"])
            alignment_right_ids[i, : ex["alignment_nb"]] = torch.LongTensor(
                ex["alignment_right_ids"]
            )

        return {
            **output,
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_nb": torch.LongTensor([ex["alignment_nb"] for ex in examples]),
        }


def keep_only_first_subword(example):
    """
    function to use in datasets.Dataset.map() for considering only the first subword
    of each word in a realignment task. This should be used simultaneously with use_first_subword_only=True
    in LabeAlignmentMapper for a token classification task
    """
    for key in ["alignment_left_positions", "alignment_right_positions"]:
        for i, (start, _) in enumerate(example[key]):
            example[key][i][1] = start + 1
    return example


class RealignmentAndOtherCollator(RealignmentCollator):
    """
    deprecated: useless with the new training loop
    Collator for building batch that contain simultaneously samples from a realignment
    task and samples for another task, handled by self.other_collator
    """

    def __init__(self, tokenizer, other_collator, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.other_collator = other_collator
        self.count_alignment = 0
        self.count_task = 0
        self.history = []

    def __call__(self, examples):
        alignment_examples = list(filter(lambda x: x.get("left_input_ids") is not None, examples))
        task_examples = list(filter(lambda x: x.get("input_ids") is not None, examples))

        self.count_alignment += len(alignment_examples)
        self.count_task += len(task_examples)

        if len(alignment_examples) > 0 and len(task_examples) > 0:
            state = "mixed"
        elif len(alignment_examples) > 0:
            state = "alignment"
        elif len(task_examples) > 0:
            state = "task"
        else:
            state = "empty"

        if len(self.history) == 0 or self.history[-1][0] != state:
            self.history.append((state, 1))
        else:
            self.history[-1] = (state, self.history[-1][1] + 1)

        if len(alignment_examples) > 0:
            try:
                alignment_batch = super(RealignmentAndOtherCollator, self).__call__(
                    alignment_examples
                )
            except Exception as e:
                raise e
        else:
            alignment_batch = {}

        if len(task_examples) > 0:
            other_inputs = [
                {
                    k: v
                    for k, v in ex.items()
                    if not k.startswith("left_")
                    and not k.startswith("right_")
                    and not k.startswith("alignment_")
                }
                for ex in task_examples
            ]
            batch_others = self.other_collator(other_inputs)
        else:
            batch_others = {}
        return {**alignment_batch, **batch_others}


def get_realignment_dataset(
    tokenizer,
    translation_dataset,
    left_lang,
    right_lang,
    dico_path=None,
    dico=None,
    mapper_for_realignment=None,
    dico_fraction=1.0,
    return_dico=False,
    first_subword_only=False,
    left_lang_id=0,
    right_lang_id=0,
    seed=None,
    max_length=None,
    ignore_identical=True,
    add_identical=False,
    split="all",
    zh_segmenter: Optional[StanfordSegmenter] = None,
):
    """
    Build a realignment dataset from a translation dataset

    Arguments:
    - tokenizer
    - translation_dataset
    - left_lang: id of the left lang (probably 'en')
    - right_lang
    - dico_path: path to the directory containing files for dictionaires (like "en-fr.txt")
    - mapper_for_realignment: None by default, can be useful if we want to define a mapper beforehand
        and filter the dictionary (train/test split) it uses for building the alignment table
    - dico_fraction: 1. by default, smaller if we want to sample the dictionary and use return_dico=True to return the test dictionary
    - return_dico: False by default, whether to return the remaining subset of the dictionary which was not used for training realignment (if dico_fraction < 1.)
    - first_subword_only: False by default, whether to realign the representation of the first subword or an average of all subwords
    - left_lang_id: an arbitrary id for the left lang (usefull only if we use orthogonal mapping in realignment)
    - right_lang_id: same for right
    - seed
    - max_length
    - ignore_identical: whether to ignore identical words in the realignment task (default to True)
    - zh_segmener: stanford segmenter for chinese
    """
    mapper = mapper_for_realignment or DatasetMapperForRealignment(
        left_lang,
        right_lang,
        dico_path=dico_path,
        dico=dico,
        dictionary_fraction=dico_fraction,
        ignore_identical=ignore_identical,
        add_identical=add_identical,
        split=split,
        zh_segmenter=zh_segmenter,
    )
    model_specific_mapper = AdaptAlignmentToTokenizerMapper(
        tokenizer, max_length=max_length, first_subword_only=first_subword_only
    )

    if not isinstance(translation_dataset, IterableDataset):
        translation_dataset = convert_dataset_to_iterable_dataset(translation_dataset)

    translation_dataset = translation_dataset.map(mapper, remove_columns=[left_lang, right_lang])

    translation_dataset = translation_dataset.map(
        lambda x: {**x, "left_lang_id": [left_lang_id], "right_lang_id": [right_lang_id]}
    )

    translation_dataset = translation_dataset.map(
        model_specific_mapper,
        remove_columns=["left_tokens", "right_tokens", "aligned_left_ids", "aligned_right_ids"],
        batched=True,
    )

    translation_dataset = (
        translation_dataset.filter(lambda x: len(x["alignment_left_ids"]) > 0)
        .shuffle(seed=seed)
        .with_format("torch")
    )
    if return_dico:
        return translation_dataset, mapper.other_dico
    return translation_dataset


def get_multilingual_news_commentary_realignment_dataset(
    tokenizer,
    lang_pairs: List[Tuple[str, str]],
    probabilities=None,
    dico_path=None,
    fastalign_path=None,
    first_subword_only=True,
    lang_to_id=None,
    dataset_name="news_commentary",
    realignment_type="dictionary",
    seed=None,
    cache_dir=None,
    max_length=None,
    ignore_identical=True,
    add_identical=False,
    split="all",
    zh_segmenter: Optional[StanfordSegmenter] = None,
):
    """
    Retrieve one or several translation datasets and transform them to create a single realignment dataset

    Arguments:
    - tokenizer
    - lang_pairs: List[Tuple[str, str]], contains tuple of languages (alpha 2 code) for getting translations datasets and dictionaries
        e.g. [("en", "fr")]
    - probabilities: probabilities associated with each pair of language for when interleaving the datasets
    - dico_path: path to the directory containing files for dictionaires (like "en-fr.txt")
    - first_subword_only: True by default, whether to realign the representation of the first subword or an average of all subwords
    - lang_to_id: None by default, dictionary which attribute an id to each language, will build one if not provided, usefull only
        if we learn an orthogonal mapping during realignment
    - dataset_name: 'news_commentary' by default, 'opus100' is also supported, desings the name of the translation dataset to use
    - seed
    - cache_dir: the datasets_cache_dir for the HF load_dataset function
    - max_length
    - ignore_identical: whether to ignore identical words in the realignment task (default to True)
    """

    if dataset_name == "news_commentary":
        dataset_getter = get_news_commentary
    elif dataset_name == "opus100":
        dataset_getter = get_opus100
    else:
        raise NotImplementedError(f"dataset_name `{dataset_name}` is not expected.")
    # by convention, we fix the pivot language as first left_lang (usually English)
    pivot = lang_pairs[0][0]
    lang_to_id = lang_to_id or {
        pivot: -1,
        **{
            lang: i
            for i, lang in enumerate(
                filter(
                    lambda x: x != pivot,
                    set(
                        list(map(lambda x: x[0], lang_pairs))
                        + list(map(lambda x: x[1], lang_pairs))
                    ),
                )
            )
        },
    }
    if realignment_type == "dictionary":
        datasets = [
            get_realignment_dataset(
                tokenizer,
                dataset_getter(left_lang, right_lang, cache_dir=cache_dir),
                left_lang,
                right_lang,
                dico_path=dico_path,
                first_subword_only=first_subword_only,
                left_lang_id=lang_to_id[left_lang],
                right_lang_id=lang_to_id[right_lang],
                seed=seed,
                max_length=max_length,
                ignore_identical=ignore_identical,
                add_identical=add_identical,
                split=split,
                zh_segmenter=zh_segmenter,
            )
            for i, (left_lang, right_lang) in enumerate(lang_pairs)
        ]
    elif realignment_type == "fastalign":
        datasets = [
            get_fastalign_realignment_dataset_from_path(
                tokenizer,
                fastalign_path,
                left_lang,
                right_lang,
                dataset_name=dataset_name,
                first_subword_only=first_subword_only,
                left_lang_id=lang_to_id[left_lang],
                right_lang_id=lang_to_id[right_lang],
                seed=seed,
                max_length=max_length,
                ignore_identical=ignore_identical,
            )
            for i, (left_lang, right_lang) in enumerate(lang_pairs)
        ]
    else:
        raise NotImplementedError(f"realignment_type `{realignment_type}` is not expected.")

    return interleave_datasets(
        list(map(infinite_iterable_dataset, datasets)), probabilities=probabilities
    )


def mix_realignment_with_dataset(
    model,
    realignment_dataset,
    task_dataset,
    n_epochs=1,
    epoch_len=None,
    label_names=None,
    strategy="during",
    seed=None,
):
    """
    deprecated: useless with the new training loop
    """
    if not isinstance(task_dataset, IterableDataset):
        epoch_len = len(task_dataset)
        iterable_task_dataset = convert_dataset_to_iterable_dataset(task_dataset, repeat=n_epochs)
    else:
        if epoch_len is not None:
            task_dataset = task_dataset.take(epoch_len)
        if n_epochs > 1:
            task_dataset = repeat_iterable_dataset(task_dataset, n_epochs)
        iterable_task_dataset = task_dataset

    features = set(next(iter(iterable_task_dataset)).keys())
    expected = set(get_signature_columns_if_needed(model, label_names))

    to_remove = list(features - expected)
    if len(to_remove) > 0:
        logging.warning(
            f"Will remove columns {to_remove} from training dataset, as they are not used as input of the model"
        )
        iterable_task_dataset = iterable_task_dataset.remove_columns(to_remove)

    if strategy == "during":
        inject_mapper = DatasetMapperForInjectingRealignmentData(realignment_dataset)
        training_dataset = iterable_task_dataset.map(inject_mapper)
    elif strategy == "before":
        training_dataset = IterableDataset(
            enumerate(
                itertools.chain(
                    realignment_dataset.take(n_epochs * epoch_len), iterable_task_dataset
                )
            )
        )
    elif strategy == "after":
        training_dataset = IterableDataset(
            enumerate(
                itertools.chain(
                    iterable_task_dataset, realignment_dataset.take(n_epochs * epoch_len)
                )
            )
        )
    else:
        raise NotImplementedError(f"Realignment strategy not implemented: {strategy}")

    return TorchCompatibleIterableDataset(training_dataset)
