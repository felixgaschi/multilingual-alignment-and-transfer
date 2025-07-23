import torch
import numpy as np
import os
from typing import List, Tuple, Optional, Set
from transformers import AutoTokenizer
from evaluate import load as load_metric
import logging
import re
from collections import defaultdict

from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter


def get_nb_layers(model):
    res = 1 + model.config.num_hidden_layers
    if hasattr(model.config, "decoder_layers"):
        res += model.config.decoder_layers + 1
    return res


def get_tokenizer_type(tokenizer):
    """
    Get the type of tokenizer (Word-piece, sentence-piece or weird other stuff)
    """
    tokens = tokenizer.tokenize("supercalificious")
    if tokens[0][0] == "▁":
        split_type = "sentencepiece"
    elif len(tokens) > 1 and tokens[1][:2] == "##":
        split_type = "wordpiece"
    elif tokens[-1][-4:] == "</w>":
        split_type = "html_like"
    else:
        raise NotImplementedError(f"Unrecognized tokenizer type: 'supercalificious' -> `{tokens}`")
    return split_type


def find_lang_key_for_mbart_like(tokenizer, lang):
    """
    Finds the id of the special token indicating the language of the sentence
    This is used in models like mBART
    (if applied to a model that do not support this, it will simply return None)
    """
    key = None
    if hasattr(tokenizer, "lang_to_code_id"):
        candidate_keys = list(
            filter(lambda x: x.split("_")[0] == lang, tokenizer.lang_to_code_id.values())
        )
        if len(candidate_keys) != 1:
            raise Exception(
                f"Could not find the right key for language `{lang}` in tokenizer lang_to_code_id: `{list(tokenizer.lang_to_code_id.values())}"
            )
        key = candidate_keys[0]
    return key


def compute_model_hidden_reprs(
    inputs, model, tokenizer, lang, device="cpu", lang_key=None, lang_id=None
):
    """
    Computes the hidden representations of a given models for given input representations
    Takes care of various model specificities
    """

    if hasattr(tokenizer, "lang2id"):
        inputs["langs"] = tokenizer.lang2id[lang] * torch.ones(
            inputs["input_ids"].size(), dtype=int
        ).to(device)

    if getattr(model, "with_mapping", False) and lang_id is not None:
        inputs["lang_id"] = lang_id * torch.ones((inputs["input_ids"].shape[0], 1), dtype=int).to(
            device
        )

    if hasattr(tokenizer, "lang_to_code_id"):
        if lang_key is None:
            raise Exception(
                "Tokenizer has `lang_to_code_id` attribute but no lang_key was provided."
            )
        inputs["input_ids"][:, 0] = tokenizer.lang_to_code_id[lang_key]

    res = model(**inputs, output_hidden_states=True)
    if hasattr(res, "encoder_hidden_states"):
        hidden_repr = res.encoder_hidden_states + res.decoder_hidden_states
    else:
        hidden_repr = res.hidden_states
    return hidden_repr


def subwordlist_to_wordlist(subwordlist: List[str], split_type="wordpiece"):
    """
    Takes a subword list typically output by the tokenize method of a Tokenizer
    and return the list of words and their start and end position in the subword list
    """
    wordlist = []
    word_positions: List[Tuple[int, int]] = []
    current_word = ""
    start_pos = 0
    for i, subword in enumerate(subwordlist):
        if split_type == "wordpiece":
            if subword[:2] == "##":
                current_word += subword[2:]
            elif len(current_word) > 0:
                wordlist.append(current_word)
                word_positions.append((start_pos, i))
                current_word = subword
                start_pos = i
            else:
                current_word = subword
        elif split_type == "sentencepiece":
            if subword[0] == "▁":
                if len(current_word) > 0:
                    wordlist.append(current_word)
                    word_positions.append((start_pos, i + 1))
                current_word = subword[1:]
                start_pos = i
            else:
                current_word += subword
        elif split_type == "html_like":
            if subword[-4:] == "</w>":
                current_word += subword[:-4]
                wordlist.append(current_word)
                word_positions.append((start_pos, i + 1))
                current_word = ""
                start_pos = i
            else:
                current_word += subword
    if len(current_word) > 0:
        wordlist.append(current_word)
        word_positions.append((start_pos, len(subwordlist)))
    return wordlist, word_positions


class UniversalTokenizer:
    def __init__(self, base_tokenizer: str = "bert-base-multilingual-uncased", cache_dir=None):
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_tokenizer, **({"cache_dir": cache_dir} if cache_dir is not None else {})
        )
        self.tokenizer_type = get_tokenizer_type(self.base_tokenizer)

    def tokenize(self, sentence, already_tokenized_by_space=False, return_offsets=False):
        """
        Optionally returns character offsets
        """
        if already_tokenized_by_space:
            tokens = sentence.split()
            offsets = None
            if return_offsets:
                offsets = []
                offset = 0
                for tok in tokens:
                    offsets.append((offset, offset + len(tok)))
                    offset += len(tok) + 1
            res = (tokens,)
            if offsets is not None:
                res += (offsets,)
            return res
        subwords = self.base_tokenizer.tokenize(sentence)
        words, subword_offsets = subwordlist_to_wordlist(subwords)
        if not return_offsets:
            return words
        offset_mapping = self.base_tokenizer(sentence, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        offsets = [
            (offset_mapping[elt[0] + 1][0], offset_mapping[elt[1]][1]) for elt in subword_offsets
        ]
        return words, offsets


class RegexTokenizer:
    """
    Tokenizer inspired by NLTK WordPunctTokenizer based on a regex
    that separates alphanumeric characters and non-alphanumeric ones
    (thanks to the wide definition of \w in Python, it should work for
    any whitespace-tokenized language and deal with punctuation)
    """

    _regexp = re.compile(r"\w+|[^\w\s]+", re.UNICODE | re.MULTILINE | re.DOTALL)

    def tokenize(self, sentence):
        return self._regexp.findall(sentence)


class ChineseTokenizer:
    """
    Tokenizer (or rather segmenter) based on the Stanford segmenter
    for chinese
    """

    def __init__(self, segmenter: StanfordSegmenter):
        self.segmenter = segmenter

    def tokenize(self, sentence):
        if not self.segmenter.entered:
            raise Exception(f"Segmenter has not started, should be used inside a with statement")
        return self.segmenter(sentence)


class ThaiTokenizer:
    def __init__(self):
        from pythainlp.tokenize import word_tokenize

        self._segmentation_fn = word_tokenize

    def tokenize(self, sentence):
        words = self._segmentation_fn(sentence)
        return list(filter(lambda x: len(x) > 0, words))


class JapaneseTokenizer:
    def __init__(self):
        from konoha import WordTokenizer
        self.processor = WordTokenizer('sudachi', mode="A")

    def tokenize(self, sentence):
        return [token.surface for token in self.processor.tokenize(sentence)]


class LanguageSpecificTokenizer:
    """
    Tokenizer that tokenized differently according to language.
    If the language is chinese (Mandarin), it uses the Stanford segmenter
    otherwise, it uses the RegexTokenizer
    """

    def __init__(self, lang=None, zh_segmenter=None):
        """
        - zh_segmenter: optional, but if not provided (and started with a with-statement), will default
        to regex tokenizer when tokenizing chinese (not recommended)
        """
        if lang=="th":
            self._tokenizer = ThaiTokenizer()
        elif lang=="zh":
            if zh_segmenter is not None:
                self._tokenizer = ChineseTokenizer(zh_segmenter)
            else:
                logging.warning(f"LanguageSpecificTokenizer for Chinese was not passed a zh_segmenter, will default to RegexTokenizer based on whitespaces.")
                self._tokenizer = RegexTokenizer()
        elif lang == "ja":
            self._tokenizer = JapaneseTokenizer()
        else:
            self._tokenizer = RegexTokenizer()

    def tokenize(self, sentence):
        return self._tokenizer.tokenize(sentence)


def load_embedding(
    fname, vocabulary: Optional[Set[str]] = None, limit=None
) -> Tuple[List[str], np.ndarray]:
    with open(fname, "r") as f:
        count, dim = map(int, next(f).strip("\n").split())
        limit = count if limit is None else limit
        words = []
        output = np.zeros(((limit if vocabulary is None else len(vocabulary)), dim))
        for line in f:
            word, vec = line.strip("\n").split(" ", 1)
            vec = np.asarray(np.fromstring(vec, sep=" ", dtype="float"))
            if vocabulary is None or word in vocabulary:
                output[len(words)] = vec
                words.append(word)
            if len(words) == (limit if vocabulary is None else len(vocabulary)):
                break
    return words, output[: len(words)]


def get_metric_fn():
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def count_lines(fname: str):
    """
    Count line in file
    """

    def _count_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with open(fname, "rb") as fp:
        c_generator = _count_generator(fp.raw.read)
        count = sum(buffer.count(b"\n") for buffer in c_generator)
    return count
