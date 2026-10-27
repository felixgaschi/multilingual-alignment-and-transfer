from typing import Optional
from dataclasses import dataclass

import torch
import logging

from multilingual_eval.models.realignment_loss import compute_loss_from_representations

@dataclass
class RealignmentOutput:
    loss: torch.Tensor
    meta_loss: torch.Tensor = None
    per_lang_losses: dict = None

def get_automodel_replacement(model_class):
    """
    Creates a wrapper class that will instantiate the model with
    model_class.from_pretrained and wrap it into a SimplifiedModelForRealignment
    """

    class FromPretrainedProxy:

        @staticmethod
        def from_pretrained(*args, **kwargs):
            simplified_kwargs = {}
            for key in [
                "nb_pairs",
                "with_mapping",
                "regularization_to_init",
                "realignment_layers"
            ]:
                if key in kwargs:
                    logging.warning(f"Argument {key} is ignored in simplified model")
                del kwargs[key]
            for key in [
                "embedding_path",
                "encoder_path",
                "decoder_path",
                "realignment_head_config",
                "realignment_layers",
                "strong_alignment",
                "realignment_temperature",
                "realignment_loss",
                "realignment_method"
            ]:
                if key in kwargs:
                    simplified_kwargs[key] = kwargs[key]
                    del kwargs[key]

            model = model_class.from_pretrained(*args, **kwargs)
            return SimplifiedModelForRealignment(model, **simplified_kwargs)

    return FromPretrainedProxy


class SimplifiedModelForRealignment(torch.nn.Module):

    def __init__(
        self,
        model,
        embedding_path: list[str] = None,
        encoder_path: Optional[list[str]] = None,
        decoder_path: Optional[list[str]] = None,
        realignment_head_config: Optional[list[int]] = None,
        realignment_layers: Optional[list[int]] = None,
        strong_alignment=True,
        realignment_temperature=0.1,
        realignment_loss="contrastive",
        realignment_method="token"
    ):
        super(SimplifiedModelForRealignment, self).__init__()

        self.model = model
        self.embedding_path = embedding_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        self.is_encoder_decoder = (
            encoder_path is not None and decoder_path is not None
        ) or getattr(self.model.config, "is_encoder_decoder", False)

        self.realignment_head_config = realignment_head_config or [self.model.config.hidden_size, 128]
        self.build_realignment_head()

        self.realignment_layers = realignment_layers or [-1]
        self.strong_alignment = strong_alignment
        self.realignment_temperature = realignment_temperature
        self.realignment_loss = realignment_loss
        self.realignment_method = realignment_method

        #self.build_layer_pointers()
        self._to_unfreeze = []


    def build_realignment_head(self):
        transformations = []
        for i, v in enumerate(self.realignment_head_config):
            if i == 0:
                transformations.append(torch.nn.Linear(self.model.config.hidden_size, v, bias=False))
            else:
                transformations.append(
                    torch.nn.Linear(self.realignment_head_config[i - 1], v, bias=False)
                )

            if i < len(self.realignment_head_config) - 1:
                transformations.append(torch.nn.ReLU())
        if len(transformations) > 0:
            self.realignment_head = torch.nn.Sequential(*transformations).to(self.model.device)
        else:
            self.realignment_head = None

    def build_layer_pointers(self):
        self.layer_pointers = []
        embedding = self.model
        for element in self.embedding_path:
            embedding = getattr(embedding, element)
        self.layer_pointers.append(embedding)
        
        if self.encoder_path:
            encoder = self.model
            for element in self.encoder_path:
                encoder = getattr(encoder, element)
            self.layer_pointers.extend(encoder)
        
        if self.decoder_path:
            decoder = self.model
            for element in self.decoder_path:
                decoder = getattr(decoder, element)
            self.layer_pointers.extend(decoder)

    def freeze_layers(self, layers: list[int]):
        for i in layers:
            for param in self.layer_pointers[i].parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    self._to_unfreeze.append(param)
    
    def unfreeze_frozen(self):
        for param in self._to_unfreeze:
            param.requires_grad = True
        self._to_unfreeze = []

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)
        if self.realignment_head:
            self.realignment_head.to(device)
        return self

    def forward(
        self,
        left_input_ids=None,
        left_attention_mask=None,
        right_input_ids=None,
        right_attention_mask=None,
        alignment_left_ids=None,
        alignment_left_positions=None,
        alignment_right_ids=None,
        alignment_right_positions=None,
        alignment_nb=None,
        alignment_left_length=None,
        alignment_right_length=None,
        lang_probs=None,
        lang_ids=None,
        meta_loss_type="micro",
        **kwargs,
    ):
        # If we're not doing realignment, we fallback to the task 
        if left_input_ids is None:
            return self.model(**kwargs)

        left_output = self.model(
            left_input_ids,
            attention_mask=left_attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            **({"decoder_input_ids": left_input_ids[:,1:]} if self.is_encoder_decoder else {}),
            **{k.split("_", 1)[1]: v for k, v in kwargs.items() if k.startswith("left_")}
        )
        right_output = self.model(
            right_input_ids,
            attention_mask=right_attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            **({"decoder_input_ids": right_input_ids[:,1:]} if self.is_encoder_decoder else {}),
            **{k.split("_", 1)[1]: v for k, v in kwargs.items() if k.startswith("right_")}
        )

        if hasattr(left_output, "hidden_states"):
            left_hidden_states = left_output.hidden_states
            right_hidden_states = right_output.hidden_states
        elif hasattr(left_output, "encoder_hidden_states"):
            left_hidden_states = left_output.encoder_hidden_states
            right_hidden_states = right_output.encoder_hidden_states

            # Note, we do not keep the embedding layer twice, hence the `[1:]`
            left_hidden_states += left_output.decoder_hidden_states[1:]
            right_hidden_states += right_output.decoder_hidden_states[1:]

        loss, meta_loss, per_lang_losses = compute_loss_from_representations(
            left_hidden_states,
            right_hidden_states,
            self.realignment_head,
            self.realignment_layers,
            strong_alignment=self.strong_alignment,
            realignment_temperature=self.realignment_temperature,
            realignment_coef=1.0,
            realignment_loss=self.realignment_loss,
            realignment_method=self.realignment_method,
            initial_model=None,
            initial_hidden_states=None,
            # alignment labels
            alignment_left_ids=alignment_left_ids, 
            alignment_left_positions=alignment_left_positions,
            alignment_right_ids=alignment_right_ids,
            alignment_right_positions=alignment_right_positions, 
            alignment_nb=alignment_nb,
            alignment_left_length=alignment_left_length, 
            alignment_right_length=alignment_right_length,
            # language probs
            lang_probs=lang_probs,
            lang_ids=lang_ids,
            meta_loss_type=meta_loss_type,
        )

        return RealignmentOutput(
            loss=loss,
            meta_loss=meta_loss,
            per_lang_losses=per_lang_losses,
        )