import shutil

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoModel,
)


from multilingual_eval.datasets.wikiann_ner import get_wikiann_ner, get_wikiann_metric_fn
from multilingual_eval.datasets.xnli import get_xnli, sentence_classifcation_metric_fn
from multilingual_eval.datasets.topic_classification import get_topic_classification
from multilingual_eval.datasets.xlsum import (
    get_xlsum,
    get_xlsum_clm,
    xlsum_metric_fn,
    xlsum_clm_metric_fn,
)
from multilingual_eval.datasets.xtreme_udpos import get_wuetal_udpos, get_xtreme_udpos, get_xtreme_r_udpos
from multilingual_eval.datasets.pawsx import get_pawsx, pawsx_metric_fn
from multilingual_eval.datasets.token_classification import get_token_classification_metrics
from multilingual_eval.datasets.xquad import get_xquad
from multilingual_eval.datasets.question_answering import (
    get_question_answering_metrics,
    get_question_answering_getter,
)

from multilingual_eval.models.with_realignment_factory import (
    AutoModelForSequenceClassificationWithRealignment,
    AutoModelForTokenClassificationWithRealignment,
    AutoModelForQuestionAnsweringWithRealignment,
)

from multilingual_eval.models.simplified import get_automodel_replacement


def get_dataset_fn(name, zh_segmenter=None):
    """
    Function that returns a function allowing to obtain a given fine-tuning dataset
    """
    return {
        "wikiann": lambda *args, **kwargs: get_wikiann_ner(
            *args, **kwargs, zh_segmenter=zh_segmenter, resegment_zh=zh_segmenter is not None
        ),
        "udpos": get_wuetal_udpos,
        "xtreme.udpos": get_xtreme_udpos,
        "xtreme_r.udpos": get_xtreme_r_udpos,
        "xnli": get_xnli,
        "sib200": get_topic_classification,
        "pawsx": get_pawsx,
        "xquad": get_xquad,
        "xlsum": get_xlsum,
        "xlsum_clm": get_xlsum_clm,
    }[name]


def get_dataset_metric_fn(name):
    return {
        "wikiann": get_wikiann_metric_fn,
        "udpos": get_token_classification_metrics,
        "xtreme.udpos": get_token_classification_metrics,
        "xtreme_r.udpos": get_token_classification_metrics,
        "xnli": lambda: sentence_classifcation_metric_fn,
        "sib200": lambda: sentence_classifcation_metric_fn,
        "pawsx": lambda: pawsx_metric_fn,
        "xquad": lambda: get_question_answering_metrics(),
        "xlsum": lambda: xlsum_metric_fn,
        "xlsum_clm": lambda: xlsum_clm_metric_fn,
    }[name]

def llama_qa_hotfix(save_dir: str):
    """
    This hotfix is necessary for handling a bug with old version of
    transformers, see:
    https://github.com/huggingface/transformers/issues/30381#issuecomment-2247122579
    """

    class LlamaHotfixPretrainedProxy:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = AutoModel.from_pretrained(*args, **kwargs)
            model.save_pretrained(save_dir)
            model = AutoModelForQuestionAnswering.from_pretrained(save_dir)
            shutil.rmtree(save_dir)
            return model

    return LlamaHotfixPretrainedProxy


def model_fn(task_name, with_realignment=False, simplified=False, llama_qa_hotfix_dir=None):
    """
    Get the model with the right head for the fine-tuning task
    and the right head for realignment
    """
    task_name = "clirmatrix" if "clirmatrix" in task_name else task_name
    if llama_qa_hotfix_dir:
        AutoModelForQuestionAnswering = llama_qa_hotfix(llama_qa_hotfix_dir)

    if with_realignment and simplified:
        token_classification = get_automodel_replacement(AutoModelForTokenClassification)
        sequence_classification = get_automodel_replacement(AutoModelForSequenceClassification)
        question_answering = get_automodel_replacement(AutoModelForQuestionAnswering)
        retrieval = get_automodel_replacement(AutoModelForTokenClassification)
        seq_to_seq = get_automodel_replacement(AutoModelForSeq2SeqLM)
        causal_lm = get_automodel_replacement(AutoModelForCausalLM)
    elif with_realignment:
        token_classification = AutoModelForTokenClassificationWithRealignment
        sequence_classification = AutoModelForSequenceClassificationWithRealignment
        question_answering = AutoModelForQuestionAnsweringWithRealignment
        retrieval = None #Not yet implemented
        seq_to_seq = None #Not yet implemented
        causal_lm = None #Not yet implemented
    else:
        token_classification = AutoModelForTokenClassification
        sequence_classification = AutoModelForSequenceClassification
        question_answering = AutoModelForQuestionAnswering
        retrieval = AutoModelForTokenClassification
        seq_to_seq = AutoModelForSeq2SeqLM
        causal_lm = AutoModelForCausalLM
    return {
        "wikiann": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=7
        ),
        "udpos": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=18
        ),
        "xtreme.udpos": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=18
        ),
        "xtreme_r.udpos": lambda *args, **kwargs: token_classification.from_pretrained(
            *args, **kwargs, num_labels=18
        ),
        "xnli": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=3
        ),
        "sib200": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=7
        ),
        "pawsx": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=3
        ),
        "xquad": lambda *args, **kwargs: question_answering.from_pretrained(*args, **kwargs),
        "clirmatrix": lambda *args, **kwargs: retrieval.from_pretrained(*args, **kwargs),
        "xlsum": lambda *args, **kwargs: seq_to_seq.from_pretrained(*args, **kwargs),
        "xlsum_clm": lambda *args, **kwargs: causal_lm.from_pretrained(*args, **kwargs),
    }[task_name]


def model_fn_with_adapter(task_name, langs=None, n_layers=1):
    from transformers.adapters import AutoAdapterModel, PfeifferInvConfig, PfeifferConfig

    # Note: contrary to models created with model_fn,
    # realignment loss is computed from outside the model
    # because I'm tired of rewriting the definitions of models
    def get_model(*args, **kwargs):
        model = AutoAdapterModel.from_pretrained(*args, **kwargs)

        if langs:
            inv_config = PfeifferInvConfig()
            for lang in langs:
                model.add_adapter(f"{lang}_adapter", config=inv_config)
                model.add_masked_lm_head(f"{lang}_adapter")

        model.add_adapter("task", config=PfeifferConfig())

        # verify the naming convention for head
        if task_name == "wikiann":
            model.add_tagging_head("task", num_labels=7, overwrite_ok=True, layers=n_layers)
        elif task_name in ["udpos", "xtreme.udpos"]:
            model.add_tagging_head("task", num_labels=18, overwrite_ok=True, layers=n_layers)
        elif task_name in ["xnli", "pawsx"]:
            model.add_classification_head("task", num_labels=3, overwrite_ok=True, layers=n_layers)
        elif task_name == "sib200":
            model.add_classification_head("task", num_labels=7, overwrite_ok=True, layers=n_layers)
        else:
            raise NotImplementedError(task_name)

        return model

    return get_model


def model_fn_from_scratch(task_name, with_realignment=False):
    if with_realignment:
        token_classification = AutoModelForTokenClassificationWithRealignment
        sequence_classification = AutoModelForSequenceClassificationWithRealignment
        question_answering = AutoModelForQuestionAnsweringWithRealignment
    else:
        token_classification = AutoModelForTokenClassification
        sequence_classification = AutoModelForSequenceClassification
        question_answering = AutoModelForQuestionAnswering
    return {
        "wikiann": lambda *args, **kwargs: token_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=7)
        ),
        "udpos": lambda *args, **kwargs: token_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=18)
        ),
        "xtreme.udpos": lambda *args, **kwargs: token_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=18)
        ),
        "xnli": lambda *args, **kwargs: sequence_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=3)
        ),
        "sib200": lambda *args, **kwargs: sequence_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=7)
        ),
        "pawsx": lambda *args, **kwargs: sequence_classification.from_config(
            AutoConfig.from_pretrained(*args, **kwargs, num_labels=3)
        ),
        "xquad": lambda *args, **kwargs: question_answering.from_config(
            AutoConfig.from_pretrained(*args, **kwargs)
        ),
    }[task_name]


from transformers import DataCollatorForTokenClassification

class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        # Extract language information and then remove it from the features
        languages = [feature.pop("language", None) for feature in features]

        # Use the parent class's method to collate the rest of the data
        batch = super().__call__(features)

        # Add the languages back to the batch
        batch["language"] = languages

        return batch


def collator_fn(task_name):
    if task_name in ["wikiann", "udpos", "xtreme.udpos", "xtreme_r.udpos"]:
        return DataCollatorForTokenClassification
    elif task_name in ["xnli", "pawsx", "xquad", "sib200"]:
        return DataCollatorWithPadding
    elif task_name in ["xlsum", "xlsum_clm"]:
        # for xlsum_clm, labels have the same length as input_ids per example,
        # so padding them separately (with -100) keeps them aligned
        return DataCollatorForSeq2Seq
    raise KeyError(task_name)
