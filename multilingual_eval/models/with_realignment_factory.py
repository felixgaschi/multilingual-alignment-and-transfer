import functools
from typing import Type, Optional, List
import copy
import torch
import logging

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    SequenceClassifierOutput,
    QuestionAnsweringModelOutput,
)
from transformers import (
    BertModel,
    XLMRobertaModel,
    DistilBertModel,
)

from merge_args import merge_args
from multilingual_eval.models.modified_transformers.token_classification import (
    CustomBertForTokenClassification,
    CustomRobertaForTokenClassification,
    CustomDistilBertForTokenClassification,
)
from multilingual_eval.models.modified_transformers.sequence_classification import (
    CustomBertForSequenceClassification,
    CustomRobertaForSequenceClassification,
    CustomDistilBertForSequenceClassification,
)
from multilingual_eval.models.modified_transformers.question_answering import (
    CustomBertForQuestionAnswering,
    CustomRobertaForQuestionAnswering,
    CustomDistilBertForQuestionAnswering,
)

from multilingual_eval.models.realignment_loss import compute_realignment_loss
from multilingual_eval.models.modified_transformers.utils import get_class_from_model_path


def model_with_realignment_factory(
    base_class: Type[BertPreTrainedModel], output_class: Type[ModelOutput]
):
    """
    Factory function for building a class for a model that can handle a specific task (e.g.
    token classification or sequence classification) as well as a realignment task
    """
    if not isinstance(base_class, type(BertPreTrainedModel)):
        logging.warn(
            f"For building a model with realignment you used a base class that does not inherit from `BertPretrainedModel`, this might fail"
        )

    class SpecificPretrainedModelWithRealignment(base_class):
        """
        Model that can be trained for the same task as base_class but also for the realignment task
        """

        def __init__(
            self,
            config,
            realignment_loss="contrastive",
            realignment_method="token",
            with_mapping=False,
            train_only_mapping: Optional[bool] = None,
            realignment_layers: Optional[List[int]] = None,
            realignment_transformation: Optional[List[int]] = None,
            strong_alignment=False,
            realignment_temperature=0.1,
            realignment_coef=1.0,
            no_backward_for_source=False,
            regularization_to_init: Optional[bool] = None,
            regularization_lambda=1.0,
            nb_pairs=1,
        ):
            super().__init__(config, with_mapping=with_mapping, nb_pairs=nb_pairs)

            self.realignment_loss = realignment_loss
            self.realignment_method = realignment_method

            assert realignment_loss in ["contrastive", "l2"]

            # Setting up default argument according to method
            realignment_layers = realignment_layers or [-1]
            if realignment_transformation is None:
                realignment_transformation = (
                    [] if realignment_loss == "l2" else [config.hidden_size, 128]
                )
            if regularization_to_init is None:
                regularization_to_init = realignment_loss == "l2" and not with_mapping
            if train_only_mapping is None:
                train_only_mapping = with_mapping

            if train_only_mapping and not with_mapping:
                raise Exception(
                    f"train_only_mapping can't be true if with_mapping isn't, because there would be nothing to train."
                )

            if isinstance(realignment_layers, int):
                realignment_layers = [realignment_layers]
            n_layers = config.num_hidden_layers
            realignment_layers = list(
                map(lambda x: x if x >= 0 else n_layers + x, realignment_layers)
            )
            self.realignment_layers = realignment_layers

            if with_mapping:
                for layer in self.realignment_layers:
                    if layer != n_layers - 1:
                        raise NotImplementedError(
                            f"Realignment with additional mapping not implemented with other layers than the last one. Got {layer}"
                        )

            transformations = []
            for i, v in enumerate(realignment_transformation):
                if i == 0:
                    transformations.append(torch.nn.Linear(config.hidden_size, v, bias=False))
                else:
                    transformations.append(
                        torch.nn.Linear(realignment_transformation[i - 1], v, bias=False)
                    )

                if i < len(realignment_transformation) - 1:
                    transformations.append(torch.nn.ReLU())
            if len(transformations) > 0:
                self.realignment_transformation = torch.nn.Sequential(*transformations)
            else:
                self.realignment_transformation = None

            self.strong_alignment = strong_alignment

            self.train_only_mapping = train_only_mapping

            self.realignment_temperature = realignment_temperature

            self.realignment_coef = realignment_coef

            self.no_backward_for_source = no_backward_for_source

            self.regularization_to_init = regularization_to_init
            self.initial_model = None

            self.regularization_lambda = regularization_lambda

        def reset_initial_weights_regularizer(self):
            """
            Resets a copy of the model to the current weights
            (usefull for Cao et al. 2020 regularization)
            """
            if not self.regularization_to_init:
                return
            self.initial_model = copy.deepcopy(self.get_encoder())

            for param in self.initial_model.parameters():
                param.requires_grad = False

        def check_usual_args(self, **usual_args):
            for v in usual_args.values():
                if isinstance(v, torch.Tensor):
                    return True
                return False

        def get_compute_realignment_loss_fn(self):
            return functools.partial(
                compute_realignment_loss,
                self.get_encoder(),
                self.realignment_transformation,
                self.realignment_layers,
                strong_alignment=self.strong_alignment,
                realignment_temperature=self.realignment_temperature,
                realignment_coef=self.realignment_coef,
                no_backward_for_source=self.no_backward_for_source,
                regularization_lambda=self.regularization_lambda,
                initial_model=self.initial_model,
                realignment_loss=self.realignment_loss,
                train_only_mapping=self.train_only_mapping,
                realignment_method=self.realignment_method,
            )

        @merge_args(base_class.forward)
        def forward(
            self,
            # sentences from left language
            left_input_ids=None,
            left_attention_mask=None,
            left_token_type_ids=None,
            left_position_ids=None,
            left_head_mask=None,
            left_inputs_embeds=None,
            left_lang_id=None,
            # sentences from right language
            right_input_ids=None,
            right_attention_mask=None,
            right_token_type_ids=None,
            right_position_ids=None,
            right_head_mask=None,
            right_inputs_embeds=None,
            right_lang_id=None,
            # alignment labels
            alignment_left_ids=None,
            alignment_left_positions=None,
            alignment_right_ids=None,
            alignment_right_positions=None,
            alignment_nb=None,
            alignment_left_length=None,
            alignment_right_length=None,
            # lang_probs for languages alignment loss contribution
            lang_probs=None,
            lang_ids=None,
            meta_loss_type="micro",
            # kwargs from the base model
            lang_id=None,
            **usual_args,
        ):

            if left_input_ids is not None:
                realignment_loss, meta_loss, _ = self.get_compute_realignment_loss_fn()(
                    left_input_ids=left_input_ids,
                    left_attention_mask=left_attention_mask,
                    left_token_type_ids=left_token_type_ids,
                    left_position_ids=left_position_ids,
                    left_head_mask=left_head_mask,
                    left_inputs_embeds=left_inputs_embeds,
                    left_lang_id=left_lang_id,
                    right_input_ids=right_input_ids,
                    right_attention_mask=right_attention_mask,
                    right_token_type_ids=right_token_type_ids,
                    right_position_ids=right_position_ids,
                    right_head_mask=right_head_mask,
                    right_inputs_embeds=right_inputs_embeds,
                    right_lang_id=right_lang_id,
                    alignment_left_ids=alignment_left_ids,
                    alignment_left_positions=alignment_left_positions,
                    alignment_right_ids=alignment_right_ids,
                    alignment_right_positions=alignment_right_positions,
                    alignment_nb=alignment_nb,
                    alignment_left_length=alignment_left_length,
                    alignment_right_length=alignment_right_length,
                    lang_probs=lang_probs,
                    lang_ids=lang_ids,
                    meta_loss_type=meta_loss_type,
                )

            labels = usual_args.get("labels")
            return_dict = usual_args.get("return_dict")
            is_usual_args_used = self.check_usual_args(**usual_args)
            if is_usual_args_used and left_input_ids is not None:
                res = super().forward(**usual_args, lang_id=lang_id)
                if labels is not None:
                    if return_dict:
                        res.loss += realignment_loss
                        return res
                    else:
                        return (res[0] + realignment_loss, *res[1:])
            elif is_usual_args_used:
                return super().forward(**usual_args, lang_id=lang_id)
            elif left_input_ids is not None: # Apply to before strat only for now
                base_return = realignment_loss if not return_dict else output_class(loss=realignment_loss)
                
                if meta_loss:
                    return (base_return, meta_loss)
                else:
                    return (base_return,) if not return_dict else base_return
            else:
                raise Exception(f"both usual argument of the model and realignment ones were empty")

    return SpecificPretrainedModelWithRealignment


BertForTokenClassificationWithRealignment = model_with_realignment_factory(
    CustomBertForTokenClassification, TokenClassifierOutput
)
BertForSequenceClassificationWithRealignment = model_with_realignment_factory(
    CustomBertForSequenceClassification, SequenceClassifierOutput
)

RobertaForTokenClassificationWithRealignment = model_with_realignment_factory(
    CustomRobertaForTokenClassification, TokenClassifierOutput
)
RobertaForSequenceClassificationWithRealignment = model_with_realignment_factory(
    CustomRobertaForSequenceClassification, SequenceClassifierOutput
)


class AutoModelForSequenceClassificationWithRealignment:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return model_with_realignment_factory(
                CustomBertForSequenceClassification, SequenceClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return model_with_realignment_factory(
                CustomRobertaForSequenceClassification, SequenceClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return model_with_realignment_factory(
                CustomDistilBertForSequenceClassification, SequenceClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"AutoModelForSequenceClassificationWithRealignment.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )


class AutoModelForTokenClassificationWithRealignment:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return model_with_realignment_factory(
                CustomBertForTokenClassification, TokenClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return model_with_realignment_factory(
                CustomRobertaForTokenClassification, TokenClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return model_with_realignment_factory(
                CustomDistilBertForTokenClassification, TokenClassifierOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"AutoModelForTokenClassificationWithRealignment.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )


class AutoModelForQuestionAnsweringWithRealignment:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return model_with_realignment_factory(
                CustomBertForQuestionAnswering, QuestionAnsweringModelOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return model_with_realignment_factory(
                CustomRobertaForQuestionAnswering, QuestionAnsweringModelOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return model_with_realignment_factory(
                CustomDistilBertForQuestionAnswering, QuestionAnsweringModelOutput
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"AutoModelForQuestionAnsweringWithRealignment.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )
