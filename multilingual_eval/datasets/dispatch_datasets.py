from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    AutoConfig,
)


from multilingual_eval.datasets.wikiann_ner import get_wikiann_ner, get_wikiann_metric_fn
from multilingual_eval.datasets.xnli import get_xnli, xnli_metric_fn
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
        "pawsx": get_pawsx,
        "xquad": get_xquad,
    }[name]


def get_dataset_metric_fn(name):
    return {
        "wikiann": get_wikiann_metric_fn,
        "udpos": get_token_classification_metrics,
        "xtreme.udpos": get_token_classification_metrics,
        "xtreme_r.udpos": get_token_classification_metrics,
        "xnli": lambda: xnli_metric_fn,
        "pawsx": lambda: pawsx_metric_fn,
        "xquad": lambda: get_question_answering_metrics(),
    }[name]


def get_model_class_for_dataset_with_realignment(name):
    return {
        "wikiann": AutoModelForTokenClassificationWithRealignment,
        "udpos": AutoModelForTokenClassificationWithRealignment,
        "xtreme.udpos": AutoModelForTokenClassificationWithRealignment,
        "xnli": AutoModelForSequenceClassificationWithRealignment,
        "pawsx": AutoModelForSequenceClassificationWithRealignment,
        "xquad": AutoModelForQuestionAnsweringWithRealignment,
    }[name]


def model_fn(task_name, with_realignment=False):
    """
    Get the model with the right head for the fine-tuning task
    and the right head for realignment
    """
    if with_realignment:
        token_classification = AutoModelForTokenClassificationWithRealignment
        sequence_classification = AutoModelForSequenceClassificationWithRealignment
        question_answering = AutoModelForQuestionAnsweringWithRealignment
    else:
        token_classification = AutoModelForTokenClassification
        sequence_classification = AutoModelForSequenceClassification
        question_answering = AutoModelForQuestionAnswering
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
        "pawsx": lambda *args, **kwargs: sequence_classification.from_pretrained(
            *args, **kwargs, num_labels=3
        ),
        "xquad": lambda *args, **kwargs: question_answering.from_pretrained(*args, **kwargs),
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
    elif task_name in ["xnli", "pawsx", "xquad"]:
        return DataCollatorWithPadding
    raise KeyError(task_name)
