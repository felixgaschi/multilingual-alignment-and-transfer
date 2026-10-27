import logging
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import DataCollatorForTokenClassification
from transformers.optimization import get_scheduler
from torch.optim import Adam, SGD
import numpy as np
import random
import math
import hashlib
import os
import glob
import re
import json
import dataclasses

from multilingual_eval.training.states import TrainingState
from multilingual_eval.training.epoch_loop import epoch_loop, weighted_selection_realignment, ucb_weighted_selection_realignment
from multilingual_eval.training.decoupled_weighted_selection import decoupled_weighted_selection_realignment
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator,
)
from multilingual_eval.training.evaluation_loops import (
    evaluate_several_token_classification,
    evaluate_token_classification,
)

from multilingual_eval.models.simplified import SimplifiedModelForRealignment

def select_layers_for_realignment(total_layers, K):
    """
    Selects K layers by dividing the model into K blocks and choosing one layer per block.

    Args:
        total_layers (int): Total number of layers in the model.
        K (int): Number of layers to select.

    Returns:
        list: Selected layer indices (sorted).
    """
    if K > total_layers:
        return list(range(total_layers))

    # Divide layers into K blocks
    block_size = total_layers // K
    blocks = []
    for i in range(K):
        start = i * block_size
        end = (i + 1) * block_size if i < K - 1 else total_layers  # Last block takes remaining layers
        blocks.append((start, end))

    # Select one layer from each block
    selected_layers = []
    for block in blocks:
        start, end = block
        if len(selected_layers) > 0 and selected_layers[-1] == start - 1:
            start += 1
        selected_layers.append(random.choice(range(start, end)))
    
    return sorted(selected_layers)

def freeze_all(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            param.requires_grad = False
    return names

def unfreeze_all(model, names):
    for name, param in model.named_parameters():
        if name in names:
            param.requires_grad = True

def get_high_ani_layers_for_realignment(model_name: str, method: str):
    """
    model_name: ["xlm-roberta-base", "bert-base-multilingual-cased", "distilbert-base-multilingual-cased"]
    method: ["mean", "half"]
    """
    import pandas as pd
    ani_pretrain_res = pd.read_csv("scripts/2025_alignfreeze_continuation/distillation/anisotropy_results/pretrain/anisotropy_pretrain_1000.csv")[lambda x: x['Model'] == model_name]
    if method == "mean":
        mean_ani = ani_pretrain_res.Avg_anisotropy_xlang.mean()
        high_ani_layers = ani_pretrain_res[lambda x: x['Avg_anisotropy_xlang'] > mean_ani].Layer.tolist()
    elif method in ["half", "onethird"]:
        ani_pretrain_res_sorted = ani_pretrain_res.sort_values(by='Avg_anisotropy_xlang', ascending=False)
        if method == "half":
            num_layers_to_return = len(ani_pretrain_res_sorted) // 2 
        else:
            num_layers_to_return = len(ani_pretrain_res_sorted) // 3 
        high_ani_layers = ani_pretrain_res_sorted['Layer'].head(num_layers_to_return).tolist()
    else:
        raise NotImplementedError(f"Method {method} is not implemented for model {model_name}")
    return sorted(high_ani_layers)

def get_ignored_prefix_weights():
    """
    return variants of prefix to ignore when loading realignment weights
    (typically task-specific head)
    """
    prefixes = ['classifier', 'score']
    variants = []
    for p in prefixes:
        variants.append(f'{p}.')
        # with SimplifiedModel
        variants.append(f'model.{p}.')
        # with adapters
        variants.append(f'base_model.model.{p}.')
        # with both
        variants.append(f'model.base_model.model.{p}.')
    return variants

# Define a function to log the status of the Embedding space and each RobertaLayer (frozen or unfrozen)
# def log_layer_status(model, model_name):
#     if isinstance(model, SimplifiedModelForRealignment):
#         model = model.model
    
#     base_model_attrs = ["roberta", "distilbert", "bert"]
#     base_model = model
#     for attr in base_model_attrs:
#         if hasattr(model, attr):
#             base_model = getattr(model, attr)
#             break
    
#     embeddings = getattr(base_model, "embeddings", None)
#     if embeddings:
#         is_unfrozen = any(p.requires_grad for p in embeddings.parameters())
#         logging.info(f"Embedding Space: {'Unfrozen' if is_unfrozen else 'Frozen'}")

#         # List of embedding sub-layers to check specifically
#         emb_map = {
#             "Word": "word_embeddings",
#             "Position": "position_embeddings",
#             "Token Type": "token_type_embeddings",
#             "LayerNorm": "LayerNorm",
#             "Dropout": "dropout"
#         }

#         for label, attr_name in emb_map.items():
#             sub_layer = getattr(embeddings, attr_name, None)
#             if sub_layer:
#                 # Handle weight check vs parameter check (for Dropout)
#                 grad_status = any(p.requires_grad for p in sub_layer.parameters()) if hasattr(sub_layer, "parameters") else "N/A"
#                 logging.info(f"{label} Embeddings: {grad_status}")
    
#     encoder = getattr(base_model, "encoder", getattr(base_model, "transformer", None))
    
#     if encoder and hasattr(encoder, "layer"):
#         for i, layer in enumerate(encoder.layer):
#             layer_unfrozen = any(p.requires_grad for p in layer.parameters())
#             logging.info(f"Layer {i}: {'Unfrozen' if layer_unfrozen else 'Frozen'}")

#     else:
#         logging.warning(f"Model type '{model_name}' not recognized. No logging performed.")

def log_layer_status(model, model_name="Model"):
    if isinstance(model, SimplifiedModelForRealignment):
        model = model.model
        
    # HF models define a `base_model_prefix` (e.g., model.bert, model.transformer)
    hf_prefix = getattr(model, "base_model_prefix", None)
    base_model = getattr(model, hf_prefix, model) if hf_prefix else model
    
    logging.info(f"========== Layer Status for: {model_name} ==========")
    
    # Storage for parameter grading status
    embeddings_grad = {}
    layers_grad = {}
    head_grad = {}
    
    # Regex to capture standard transformer layer naming variations:
    # Matches: .layer.0., .h.1., .block.2., .layers.11.
    layer_regex = re.compile(r'\.(layer|h|block|layers)\.(\d+)\.')
    
    for name, param in model.named_parameters():
        requires_grad = param.requires_grad
        
        # Embedding layers (Word Token Embedding (WTE) and Word Position Embedding (WPE))
        if "embed" in name.lower() or "wte" in name or "wpe" in name:
            sub_name = name.split("embeddings.")[-1] if "embeddings." in name else name.split(".")[-2]
            embeddings_grad[sub_name] = requires_grad
            continue
            
        # Transformer blocks (Encoder/Decoder layers)
        match = layer_regex.search(name)
        if match:
            layer_type = match.group(1)
            layer_idx = int(match.group(2))
            
            prefix = "Encoder " if "encoder" in name.lower() else ("Decoder " if "decoder" in name.lower() else "")
            layer_key = f"{prefix}Layer {layer_idx}"
            
            if layer_key not in layers_grad:
                layers_grad[layer_key] = []
            layers_grad[layer_key].append(requires_grad)
            continue
            
        # Everything else (Heads, Poolers, Task Specific Layers)
        parts = name.split('.')
        component_key = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
        if component_key not in head_grad:
            head_grad[component_key] = []
        head_grad[component_key].append(requires_grad)

    if embeddings_grad:
        logging.info("--- Embedding Space ---")
        for emb_name, grad in embeddings_grad.items():
            logging.info(f"  {emb_name}: {'Unfrozen' if grad else 'Frozen'}")
            
    if layers_grad:
        logging.info("--- Transformer Blocks ---")
        
        def sort_layers(item):
            nums = re.findall(r'\d+', item[0])
            idx = int(nums[0]) if nums else 0
            priority = 0 if "Encoder" in item[0] else 1
            return (priority, idx)
            
        for layer_name, grads in sorted(layers_grad.items(), key=sort_layers):
            if all(grads):
                status = "Unfrozen"
            elif not any(grads):
                status = "Frozen"
            else:
                # PEFT/LoRA
                trainable_count = sum(grads)
                status = f"Partially Trainable ({trainable_count}/{len(grads)} parameters active)"
            logging.info(f"  {layer_name}: {status}")
            
    if head_grad:
        logging.info("--- Task Heads / Miscellaneous Layers ---")
        for comp_name, grads in head_grad.items():
            status = "Unfrozen" if all(grads) else ("Frozen" if not any(grads) else "Partially Trainable")
            logging.info(f"  {comp_name}: {status}")

    logging.info("=" * 45)

def freeze_model_using_strategy(strategy, model_name, model):
    if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print('Freezing first 6 encoders...')
        for i in range(6):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        freeze_realign_unfreeze_layers = 6
        print(f'Freezing last {freeze_realign_unfreeze_layers} encoders...')
        
        total_layers = len(model.roberta.encoder.layer)
        for i in range(total_layers - freeze_realign_unfreeze_layers, total_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} transformer blocks...')
        for i in range(layers_to_freeze):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    
    if strategy == "freeze_realign_unfreeze" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} encoder blocks...')
        for i in range(layers_to_freeze):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} encoder blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    
    if strategy == "freeze_realign_unfreeze_last_half" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.roberta.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "freeze_attn":
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            attention_structure = [layer.attention for layer in model.roberta.encoder.layer]
        elif "distilbert" in model_name:
            attention_structure = [layer.attention for layer in model.distilbert.transformer.layer]
        elif model_name.startswith("bert"):
            attention_structure = [layer.attention for layer in model.bert.encoder.layer]
        else:
            raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")

        print(f'Freezing attention structure in transformer blocks...')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters BEFORE freezing: {trainable_params}")
        for layer_attn in attention_structure:
            for param in layer_attn.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters AFTER freezing: {trainable_params}")
        print('Freezing done...')

    if "freeze_ffn" in strategy:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            ffn_structure = [
                dense for layer in model.roberta.encoder.layer
                for dense in (layer.intermediate.dense, layer.output.dense)
            ]
        elif "distilbert" in model_name:
            ffn_structure = [
                lin for layer in model.distilbert.transformer.layer
                for lin in (layer.ffn.lin1, layer.ffn.lin2)
            ]
        elif model_name.startswith("bert"):
            ffn_structure = [
                dense for layer in model.bert.encoder.layer
                for dense in (layer.intermediate.dense, layer.output.dense)
            ]
        else:
            raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")

        print(f'Freezing FFN structure in transformer blocks...')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters BEFORE freezing: {trainable_params}")
        for layer_ffn in ffn_structure:
            for param in layer_ffn.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters AFTER freezing: {trainable_params}")
        print('Freezing done...')

    if re.match(r"high_anisotropy_.+", strategy) and not re.search(r"(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
            
        method = strategy.split("_")[-1]
        layer_to_realign = get_high_ani_layers_for_realignment(model_name, method)
        logging.info(f"High anisotropy layers to be realign: {str(layer_to_realign)}")
        for i, layer in enumerate(layers):
            if i not in layer_to_realign:
                for param in layer.parameters():
                    param.requires_grad = False

    if re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        *_, first_layer, last_layer = strategy.split("_")
        first_layer = int(first_layer)
        last_layer = int(last_layer)
        if first_layer == 0:
            if "roberta" in model_name:
                embeddings = model.roberta.embeddings
            elif "distilbert" in model_name:
                embeddings = model.distilbert.embeddings
            elif model_name.startswith("bert"):
                embeddings = model.bert.embeddings
            else:
                raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
            for param in embeddings.parameters():
                param.requires_grad = False
            first_layer = 1
        for i in range(first_layer - 1, last_layer - 1):
            if "roberta" in model_name:
                layers = model.roberta.encoder.layer
            elif "distilbert" in model_name:
                layers = model.distilbert.transformer.layer
            elif model_name.startswith("bert"):
                layers = model.bert.encoder.layer
            else:
                raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
            for param in layers[i].parameters():
                param.requires_grad = False

    if re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        *_, first_layer, last_layer = strategy.split("_")
        first_layer = int(first_layer)
        last_layer = int(last_layer)
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type /before_realign_only_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
        for i, layer in enumerate(layers):
            if i < first_layer or i >= last_layer:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    # Realign but choose K layers randomly
    k_layers = None
    if re.match(r"before_random_realign_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        *_, k_layers = strategy.split("_")
        k_layers = int(k_layers)
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type /before_realign_only_random_[0-9]+/ is not implemented for model {model_name}")
        selected_layers = select_layers_for_realignment(len(layers), k_layers)
        logging.info(f"Current selected layers! {str(selected_layers)}")
        for i, layer in enumerate(layers):
            if i not in selected_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    if re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
        
        #Extract configg
        match = re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy)
        num_realign_lays, noembs_flag, position_type = match.groups()

        selected_indices = []
        start_idx = 0
        if noembs_flag:
            if noembs_flag == "_noembs":
                layers = layers[1:] # Skip the embs layer
            elif noembs_flag == "withembs":
                selected_indices = [0] # Add the embs layer
                start_idx = 1

        #Get number of layers to freeze
        if num_realign_lays == "half":
            num_realign_lays = len(layers) // 2 
        elif num_realign_lays == "twothird":
            num_realign_lays = (2 * len(layers)) // 3    
        elif num_realign_lays == "onethird":
            num_realign_lays = len(layers) // 3
        elif num_realign_lays == "fivesixth":
            num_realign_lays = (5 * len(layers)) // 6    
        elif num_realign_lays == "onesixth":
            num_realign_lays = len(layers) // 6

        if noembs_flag == "withembs":
            num_realign_lays -= 1

        total_layers = len(layers)
        if position_type:
            if position_type == "_adjacent":
                #Select adjacent layers
                start = random.randint(start_idx, total_layers - num_realign_lays)
                selected_indices.extend(list(range(start, start + num_realign_lays)))
            elif position_type == "_discrete":
                #Select discrete layers
                possible_indices = list(range(total_layers))
                for i in range(num_realign_lays):
                    index = random.choice(possible_indices)
                    selected_indices.append(index)
                    possible_indices = [idx for idx in possible_indices if idx not in {index, index + 1, index - 1}]
                selected_indices = sorted(selected_indices)
        else:
            selected_indices.extend(sorted(random.sample(range(total_layers), num_realign_lays)))
        logging.info(f"Layers to be realign: {str(selected_indices)}")
        
        #Freezing 
        for i, layer in enumerate(layers):
            if i not in selected_indices:
                for param in layer.parameters():
                    param.requires_grad = False

    if re.search(r"realign_specific_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type {strategy} is not implemented for model {model_name}")
        
        #Extract indices
        match = re.search(r"realign_specific_[0-9]+", strategy)
        _, selected_indices = match.group(0).split("_")
        selected_indices = sorted([int(i) for i in selected_indices])
        logging.info(f"Layers to be realign: {str(selected_indices)}")
        
        #Freezing 
        for i, layer in enumerate(layers):
            if i not in selected_indices:
                for param in layer.parameters():
                    param.requires_grad = False
    
    if strategy in ["during_freeze_realign_unfreeze", 
                    "baseline_freeze_realign_unfreeze"] and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print('Freezing first 6 encoders...')
        for i in range(6):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy in ["baseline_freeze_realign_unfreeze_last_6",
                    "during_freeze_realign_unfreeze_last_6"] and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        total_layers = len(model.roberta.encoder.layer)
        for i in range(total_layers - 6, total_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    if strategy == "during_freeze_realign_unfreeze" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} transformer blocks...')
        for i in range(layers_to_freeze):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Freezing first {layers_to_freeze} encoder blocks...')
        for i in range(layers_to_freeze):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

    if strategy == "during_freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} encoder blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')
    
    if strategy == "during_freeze_realign_unfreeze_last_half" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.roberta.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Freezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print('Freezing done...')

def unfreeze_model_with_strategy(strategy, model_name, model):
    if strategy == "freeze_realign_unfreeze" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print('Unfreezing first 6 encoders...')
        for i in range(6):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    
    if strategy == "freeze_realign_unfreeze_last_6" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print('Unfreezing last 6 encoders...')
        
        total_layers = len(model.roberta.encoder.layer)
        for i in range(total_layers - 6, total_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    if strategy == "freeze_realign_unfreeze" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Unfreezing first {layers_to_freeze} transformer blocks...')
        for i in range(layers_to_freeze):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    if strategy == "freeze_realign_unfreeze_last_half" and "distilbert" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.distilbert.transformer.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Unfreezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')
    
    if strategy == "freeze_realign_unfreeze" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Freezing the first half of the layers

        print(f'Unfreezing first {layers_to_freeze} encoder blocks...')
        for i in range(layers_to_freeze):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    if strategy == "freeze_realign_unfreeze_last_half" and model_name.startswith("bert"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.bert.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Unfreezing last {layers_to_freeze} encoder blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    if strategy == "freeze_realign_unfreeze_last_half" and "roberta" in model_name:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        num_layers = len(model.roberta.encoder.layer)
        layers_to_freeze = num_layers // 2  # Number of layers to freeze

        # Calculate the starting index for freezing (freezing the last half of the layers)
        start_freezing_from_layer = num_layers - layers_to_freeze

        print(f'Unfreezing last {layers_to_freeze} transformer blocks...')
        for i in range(start_freezing_from_layer, num_layers):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = True

        print('Unfreezing done...')

    if strategy == "freeze_attn":
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print(f'Unfreezing attention structure in transformer blocks...')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters BEFORE unfreezing: {trainable_params}")
        for layer_attn in attention_structure:
            for param in layer_attn.parameters():
                param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters AFTER unfreezing: {trainable_params}")
        print('Unfreezing done...')

    if "freeze_ffn" in strategy:
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print(f'Unfreezing FFN structure in transformer blocks...')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters BEFORE unfreezing: {trainable_params}")
        for layer_ffn in ffn_structure:
            for param in layer_ffn.parameters():
                param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"___Total trainable parameters AFTER unfreezing: {trainable_params}")
        print('Unfreezing done...')

    if re.match(r"high_anisotropy_.+", strategy) and not re.search(r"(gradual|oneatatime)_(topdown|bottomup|random)_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        print(f"{str(layer_to_realign)} have been realigned. Unfreezing others...")
        for i, layer in enumerate(layers):
            if i not in layer_to_realign:
                for param in layer.parameters():
                    param.requires_grad = True
        print('Unfreezing done...')
    
    if re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        *_, first_layer, last_layer = strategy.split("_")
        first_layer = int(first_layer)
        last_layer = int(last_layer)
        if first_layer == 0:
            if "roberta" in model_name:
                embeddings = model.roberta.embeddings
            elif "distilbert" in model_name:
                embeddings = model.distilbert.embeddings
            elif model_name.startswith("bert"):
                embeddings = model.bert.embeddings
            else:
                raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
            for param in embeddings.parameters():
                param.requires_grad = True
            first_layer = 1
        for i in range(first_layer - 1, last_layer - 1):
            if "roberta" in model_name:
                layers = model.roberta.encoder.layer
            elif "distilbert" in model_name:
                layers = model.distilbert.transformer.layer
            elif model_name.startswith("bert"):
                embeddings = model.bert.encoder.layer
            else:
                raise NotImplementedError(f"Strategy of type /freeze_realign_unfreeze_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
            for param in layers[i].parameters():
                param.requires_grad = True
    
    if re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        # TODO: Create new strategy
        *_, first_layer, last_layer = strategy.split("_")
        first_layer = int(first_layer)
        last_layer = int(last_layer)
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type /before_realign_only_[0-9]+_[0-9]+/ is not implemented for model {model_name}")
        for i, layer in enumerate(layers):
            if i < first_layer or i >= last_layer:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    # Realign but choose K layers randomly
    if re.match(r"before_random_realign_[0-9]+", strategy):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")
        if "roberta" in model_name:
            layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
        elif "distilbert" in model_name:
            layers = [model.distilbert.embeddings] + list(model.distilbert.transformer.layer)
        elif model_name.startswith("bert"):
            layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        else:
            raise NotImplementedError(f"Strategy of type /before_realign_only_random_[0-9]+/ is not implemented for model {model_name}")
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = True

    if re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy) or re.search(r"realign_specific_[0-9]+", strategy):   
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")             
        logging.info(f"Unfreezing other layers than {str(selected_indices)}")
        for i, layer in enumerate(layers):
            if i not in selected_indices:
                for param in layer.parameters():
                    param.requires_grad = True
        print('Unfreezing done...')

def load_ckpt_from_path(
    ckpt_type, 
    checkpoint_path, 
    seed, 
    model_slug, 
    task_name, 
    model, 
    use_adapter, 
    adapter_approach,
    adapter_config,
    optimizer=None,
    scheduler=None,
    realignment_steps_before=None
    ):
    import peft


    VALID_TYPES = {"realignment", "finetuning"}
    if ckpt_type not in VALID_TYPES:
        raise ValueError(f"Invalid checkpoint type: {ckpt_type}. Must be in {VALID_TYPES}")

    ckpt, checkpoint_progress = None, None

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None, None

    # ==========================================================
    # REALIGNMENT CKPT
    # ==========================================================
    if ckpt_type == "realignment":

        prefix = f"{ckpt_type}_{model_slug}_seed_{seed}"

        # ------------------------------------------------------
        # 1. Try loading exact checkpoint first
        # ------------------------------------------------------
        ckpt_file = None

        if realignment_steps_before is not None:
            target_iter = realignment_steps_before - 1

            exact_ckpt = os.path.join(
                checkpoint_path,
                f"{prefix}_iter_{target_iter}.ckpt"
            )

            if os.path.exists(exact_ckpt):
                ckpt_file = exact_ckpt

                match = re.search(r"_iter_([0-9]+)\.ckpt$", ckpt_file)
                checkpoint_progress = int(match.group(1)) if match else None
                
                logging.info(f"Found exact checkpoint: {ckpt_file}")

        # ------------------------------------------------------
        # 2. Fallback: load latest valid checkpoint
        # ------------------------------------------------------
        if ckpt_file is None:

            pattern = os.path.join(
                checkpoint_path,
                f"{prefix}_iter_*.ckpt"
            )

            matching_files = glob.glob(pattern)

            if not matching_files:
                logging.warning(f"No checkpoint found for pattern: {pattern}")
                return None, prefix, None

            valid_ckpts = []

            for path in matching_files:

                match = re.search(r"_iter_([0-9]+)\.ckpt$", path)

                if not match:
                    continue

                iter_num = int(match.group(1))

                # --------------------------------------------------
                # Only allow checkpoints BEFORE target iteration
                # --------------------------------------------------
                if (
                    realignment_steps_before is None
                    or iter_num < realignment_steps_before
                ):
                    valid_ckpts.append((iter_num, path))

            if not valid_ckpts:
                logging.warning(
                    f"No valid checkpoint found before "
                    f"iteration {realignment_steps_before}"
                )
                return None, prefix, None

            # Largest valid iteration
            checkpoint_progress, ckpt_file = max(
                valid_ckpts,
                key=lambda x: x[0]
            )

            logging.warning(
                f"Exact checkpoint not found. "
                f"Loading closest previous checkpoint instead: "
                f"{ckpt_file}"
            )

    # ==========================================================
    # FINETUNING CKPT
    # ==========================================================
    else:

        iter_num = str(realignment_steps_before - 1) if realignment_steps_before else "*"

        prefix = f"{ckpt_type}_{model_slug}_{task_name}_seed_{seed}_reiter_{iter_num}"

        pattern = os.path.join(
            checkpoint_path,
            f"{prefix}_epoch_*_iter_*.ckpt"
        )

        matching_files = glob.glob(pattern)

        if not matching_files:
            return None, prefix, None

        ckpt_file = sorted(matching_files)[-1]

        match = re.search(r"epoch_([0-9]+)", ckpt_file)
        checkpoint_progress = int(match.group(1)) if match else None

    # ==========================================================
    # LOAD CHECKPOINT
    # ==========================================================
    try:

        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        is_adapter_checkpoint = ckpt.get("use_adapter", False)

        state_dict = ckpt.pop("model_state_dict", {})

        if not state_dict:
            return None, prefix, None

        # ------------------------------------------------------
        # Filter ignored weights for realignment
        # ------------------------------------------------------
        if ckpt_type == "realignment":
            ignored_prefixes = tuple(get_ignored_prefix_weights())
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith(ignored_prefixes)
            }

        # ------------------------------------------------------
        # Prefix handling
        # ------------------------------------------------------
        first_key = next(iter(state_dict.keys()), "")
        ckpt_has_prefix = first_key.startswith("model.")
        is_simplified = isinstance(model, SimplifiedModelForRealignment)

        if is_simplified and not ckpt_has_prefix:
            state_dict = {
                f"model.{k}": v for k, v in state_dict.items()
            }

        elif not is_simplified and ckpt_has_prefix:
            state_dict = {
                k[6:]: v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False
        )

        # ------------------------------------------------------
        # Load optimizer state
        # ------------------------------------------------------
        if optimizer is not None and "optimizer_state_dict" in ckpt:

            try:
                optimizer.load_state_dict(
                    ckpt["optimizer_state_dict"]
                )

                logging.info(
                    f"Optimizer state loaded from {ckpt_file}"
                )

            except Exception as e:
                logging.warning(
                    f"Failed to load optimizer state dict: {e}"
                )

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:

            try:
                scheduler.load_state_dict(
                    ckpt["scheduler_state_dict"]
                )

                logging.info(
                    f"Scheduler state loaded from {ckpt_file}"
                )

            except Exception as e:
                logging.warning(
                    f"Failed to load scheduler state dict: {e}"
                )

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------
        if is_adapter_checkpoint and use_adapter:
            # Check approach
            ckpt_adapter_approach = ckpt.get("adapter_approach", None)
            if ckpt_adapter_approach != adapter_approach:
                raise Exception(f"Checkpoint's adapter approach {ckpt_adapter_approach} does not match target approach{adapter_approach}")

            # Check config
            ckpt_adapter_config = ckpt.get("adapter_config", None)
            if ckpt_adapter_config is None or adapter_config is None:
                raise Exception(f"Checkpoint's adapter config mismatch!")
            ckpt_adapter_config_dict = ckpt_adapter_config.to_dict()
            adapter_config_dict = adapter_config.to_dict()
            all_keys = {"lora_r", "lora_alpha", "lora_dropout", "target_modules", "lora_bias"}

            for key in all_keys:
                val_loaded = ckpt_adapter_config_dict.get(key)
                val_request = adapter_config_dict.get(key)
                
                if val_loaded != val_request:
                    raise Exception(
                        f"Checkpoint's adapter config mismatch!\n"
                        f"{key}: Loaded {val_loaded} instead of {val_request}"
                    )

            logging.info(f"Adapter checkpoint loaded: {ckpt_file}")

            adapter_keys_in_ckpt = any(
                "lora" in k or "adapter" in k
                for k in state_dict.keys()
            )

            if adapter_keys_in_ckpt:

                missing_adapter_keys = [
                    k for k in missing_keys
                    if "lora" in k or "adapter" in k
                ]

                if missing_adapter_keys:
                    logging.warning(
                        f"Missing adapter keys: {missing_adapter_keys}"
                    )

        else:

            logging.info(
                f"{ckpt_type.capitalize()} checkpoint loaded: {ckpt_file}"
            )

            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")

            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")

        del state_dict

    except Exception as e:

        logging.error(
            f"Unable to load {ckpt_type} ckpt from "
            f"{ckpt_file}. Error: {e}"
        )

        return None, prefix, None

    return ckpt, prefix, checkpoint_progress


def realignment_training_loop(
    tokenizer,
    model,
    task_dataset: DataLoader,
    realignment_dataset: DataLoader or list,
    strategy="during",
    evaluation_datasets=None,
    same_language_evaluation_dataset=None,
    evaluation_prefixes=None,
    task_batch_size=4,
    nb_realignment_steps_before=None,
    extra_realignment_steps_checkpoints=None,
    realignment_batch_size=2,
    learning_rate=2e-5,
    n_epochs=10,
    accumulation_steps=1,
    ### Parameters for weighted sampling
    enable_weighted_sampling=False,
    weighted_sampling_method="meta_learning",
    decouple_meta_and_model_updates="joint",
    meta_loss_type="micro",
    meta_learning_rate=1e-2,
    inner_batches_before_outer=5,
    lbsmoothing_eps=0,
    with_regularization=False,
    lambda_entropy=1e-3,
    noise_mixing_strat=None,
    softmax_temp=1,
    ucb_exploration_coef=1.0,
    static_meta_weights_jsonl=None,
    ###
    logging_steps=100,
    log_in_wandb=False,
    result_store=None,
    metric_fn=None,
    realignment_coef=0.1,
    realignment_coef_scheduler=None,
    data_collator=None,
    seed=None,
    epoch_callbacks=None,
    realignment_step_callbacks=None,
    hash_args=None,
    cache_dir=None,
    return_model_hash=False,
    final_prefix="final",
    pretrained_model_fn=None,
    realignment_steps_by_finetuning=1,
    label_key="labels",
    model_name=None,
    checkpoint_path=None,
    task_name=None,
    noaligner=False,
    use_adapter=False,
    adapter_approach="same",
    adapter_config=None,
    always_frozen_params=None,
):
    """
    Performs a training loop, with or without realignment

    Arguments:

    - tokenizer
    - model
    - task_dataset: training dataset for the fine-tuning task (must have a length)
    - realignment_dataset: iterable dataset for the realignment auxiliary task
    - strategy: default "during", the realignment strategy (either "baseline" for no realignment, or "after", "before" or "during")
    - evaluation_datasets: optional list of evaluation datasets
    - same_language_evaluation_dataset: optional evaluation dataset on same language as training
    - evaluation_prefixes: optional list of prefixes for evaluation datasets metrics
    - task_batch_size: batch size for the training task (not considering accumulation steps)
    - nb_realignment_steps_before: if set, number of realignment batches to see before fine-tuning, otherwise it is n_epochs times the number of fine-tuning batch.
        Only taken into account if strategy is "before", "before+during" or "after"
    - realignment_batch_size: batch size of the realignment step
    - learning_rate: learning rate for both fine-tuning and realignment
    - n_epochs: number of epochs for the fine-tuning task
    - accumulation_steps: number of accumulation steps for the fine-tuning task
    - logging_steps: int, default 100, number of steps (in term of optimization steps, hence nb of batch / accumulation steps) between each log of training stats
    - log_in_wandb: (deprecated) whether to log training stats in wandb (conditional import), better to use result_store with loggers.WandbResultStore
    - result_store: an instance of loggers.DefaultResultStore that captures the different results obtained along the training (by default it logs them to the console, but it can store them in
        a dictionary for later retrieval)
    - metric_fn: function that gets the metric from the overall predictions and labels
    - realignment_coef: float, default 0.1, the coefficient to apply to the realignment loss
    - realignment_coef_scheduler: a function that takes an integer (the epoch) and return a float, the coefficient to apply to the realignment loss at
        given epochs, overrides realignment_coef
    - data_collator: default None, if None, will default to DataCollatorForTokenClassification(tokenizer)
    - seed
    - epoch_callbacks: (deprecated) optional list of function that takes the model as input and will be called before the first fine-tuning epoch and after each one
    - realignment_step_callbacks: (deprecated) like epoch_callbacks but for each realignment step
    - hash_args: (deprecated) default None, optional string to add to hashing realigned models (only with before strategy), will cache only if it is provided (ideally with model name, id for realignment dataset and commit hash)
        and if cache_dir is provided
    - cache_dir: default None, optional directory for caching models
    - return_model_hash: default False, whether to return the model hash for model saved after realignment (not fine-tuning !!!) useful only if hash_args and cache_dir are specified and if strategy == "before"
    - final_prefix: prefix for metrics in the final evaluation
    - pretrained_model_fn: (deprecated) when the model is cached, function to instantiate the pretrained model from cache_path
    - realignment_steps_by_finetuning: number of realignment optimization steps to perform by fine-tuning steps (useful in 'during' strategy)
    - label_key: (deprecated) the key for the labels in the training input
    """

    def get_optimizer():
        # if "gemma" not in model_name:
        #     return Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        # else:
        #     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #     print("Changed from Adam to SGD for Gemma")
        #     logging.info("Changed from Adam to SGD for Gemma")
        #     return SGD(model.parameters(), lr=learning_rate*10, momentum=0.9, weight_decay=0.01, nesterov=True)
        return Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # Creates a path friendly slug for the model name
    model_slug = model_name.replace("/", "__")
    if use_adapter:
        model_slug += "_with_adapter"
        if adapter_approach == "same":
            pass
        elif adapter_approach == "separate":
            model_slug += "_separate"
        elif adapter_approach == "realign_only":
            model_slug += "_realign_only"
        else:
            raise NotImplementedError(f"Adapter approach {adapter_approach} is not implemented.")

    if use_adapter and adapter_approach in ["separate", "realign_only"] and strategy not in ["baseline", "before"]:
        raise NotImplementedError(f"Using separate adapters is only implemented for baseline and before strategies. Got {strategy}.")

    if enable_weighted_sampling:
        if strategy not in ["before"]:
            raise NotImplementedError(f"Using weighted sampling is only implemented for before strategies. Got {strategy}.")
        VALID_WEIGHTED_SAMPLING_METHODS = ["meta_learning", "ucb"]
        if weighted_sampling_method not in VALID_WEIGHTED_SAMPLING_METHODS:
            raise NotImplementedError(f"Unknown weighted_sampling_method: {weighted_sampling_method}. Choose from {VALID_WEIGHTED_SAMPLING_METHODS}.")
        if weighted_sampling_method == "meta_learning":
            VALID_META_LOSS_TYPE = ["micro", "macro", "micro_log"]
            if meta_loss_type not in VALID_META_LOSS_TYPE:
                raise NotImplementedError(f"Meta loss type {meta_loss_type} not implemented. Avaiable type: {VALID_META_LOSS_TYPE}")

    data_collator = data_collator or DataCollatorForTokenClassification(tokenizer)
    epoch_callbacks = epoch_callbacks or []
    realignment_step_callbacks = realignment_step_callbacks or []

    if log_in_wandb:
        import wandb

    # Put model to GPU if available
    if model.device.type != "cuda" and torch.cuda.device_count() > 0:
        model = model.to(0)

    # Fix random seed for Pytorch and numpy
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        g = None
        seed_worker = None

    # Create dataloader for the fine-tuning task
    task_dataloader = DataLoader(
        task_dataset,
        shuffle=True,
        batch_size=task_batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    ) if task_dataset else None

    print()
    print(f'RUNNING MODEL: {model_name}')
    print(model)
    print()

    # If needed, create dataloader for re-alignment task
    if strategy.find('baseline') == -1:
        # Note: if this line is modified, hashing args for caching must be checked

        print("Not Using Baseline")
        print()

        realignment_dataloader = DataLoader(
            realignment_dataset,
            shuffle=False,
            batch_size=realignment_batch_size,
            collate_fn=RealignmentAndOtherCollator(
                tokenizer,
                data_collator,
                noaligner=noaligner,
            ),
        ) if not enable_weighted_sampling else None # Weighted sampling requires data iterator for each language.
    else:
        print("Using Baseline")
        realignment_dataloader = None

    training_state = TrainingState.compute_expected_samples(
        strategy,
        task_dataset,
        task_dataloader,
        n_epochs,
        task_batch_size,
        realignment_batch_size,
        accumulation_steps=accumulation_steps,
        nb_realignment_steps_before=nb_realignment_steps_before,
    )

    # If available, create dataloader for evaluation on training language
    if same_language_evaluation_dataset is not None:
        same_language_evaluation_dataloader = DataLoader(
            same_language_evaluation_dataset,
            shuffle=False,
            batch_size=task_batch_size,
            collate_fn=data_collator,
        )

    use_caching = False
    selected_layers = None

    # Check for existing realignment ckpt
    realignment_steps_before = (
            math.ceil(len(task_dataloader) / accumulation_steps) * n_epochs
            if nb_realignment_steps_before is None
            else nb_realignment_steps_before
        )
    
    # Note: if this line is modified, hashing args for caching must be checked
    before_optimizer = get_optimizer()
    realignment_ckpt, realignment_checkpoint_prefix_name, ckpt_iter = load_ckpt_from_path(
        ckpt_type="realignment",
        checkpoint_path=checkpoint_path, 
        seed=seed, 
        model_slug=model_slug, 
        task_name=task_name, 
        model=model, 
        use_adapter=use_adapter, 
        adapter_approach=adapter_approach,
        adapter_config=adapter_config,
        optimizer=before_optimizer,
        realignment_steps_before=nb_realignment_steps_before
        )

    # If strategy is "before" or "before+during", perform realignment before fine-tuning
    if strategy in ["before", "before+during", 
                    "freeze_realign_unfreeze",
                    "freeze_realign_unfreeze_last_half",
                    "freeze_realign_unfreeze_last_6",
                    "freeze_attn",
                    "freeze_ffn",
                   ] or re.match(r"freeze_realign_unfreeze_[0-9]+_[0-9]+", strategy) or re.match(r"before_realign_only_[0-9]+_[0-9]+", strategy) \
                       or re.match(r"before_random_realign_[0-9]+", strategy) \
                           or re.match(r"before_gradual_(topdown|bottomup|random)_[0-9]+", strategy) \
                               or re.match(r"before_oneatatime_(topdown|bottomup|random)_[0-9]+", strategy) \
                                    or re.match(r"high_anisotropy_.+", strategy) \
                                        or re.search(r"realign_random_(half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?", strategy) \
                                            or re.search(r"realign_specific_[0-9]+", strategy) \
                                                or "freeze_ffn" in strategy:

        use_caching = cache_dir is not None and hash_args is not None and seed is not None

        learning_rate = learning_rate

        if use_caching:
            string_to_hash = (
                hash_args
                + "__"
                + "__".join(
                    [
                        str(learning_rate),
                        str(seed),
                        str(realignment_batch_size),
                        str(realignment_steps_before),
                    ]
                )
            )
            model_hash = hashlib.md5(string_to_hash.encode()).hexdigest()

            cache_path = os.path.join(cache_dir, model_hash)
            training_state_path = os.path.join(cache_dir, f"{model_hash}.json")
            info_path = os.path.join(cache_dir, f"{model_hash}.info")
        else:
            cache_path = None
            training_state_path = None
            info_path = None

        found_caching = False

        if cache_path is not None and os.path.isfile(training_state_path):
            try:
                with open(training_state_path, "r") as f:
                    other_training_state = TrainingState(**json.load(f))
                    training_state.update_from_other_finetuning(other_training_state)
                found_caching = True
            except json.decoder.JSONDecodeError:
                logging.error(f"Could not decode cached training state. Will not use the cache.")

        if found_caching:
            logging.info(f"Loading cached model: {model_hash}")
            if isinstance(model, SimplifiedModelForRealignment):
                raise Exception("SimplifiedModelForRealignment is not compatible with custom caching")
            model = (
                pretrained_model_fn(cache_path, ignore_mismatched_sizes=True)
                if pretrained_model_fn is not None
                else model.__class__.from_pretrained(cache_path, ignore_mismatched_sizes=True)
            )
        else:

            # print()
            # print('Realignment Dataloader')
            # for i, batch in enumerate(realignment_dataloader):
            #     # print(batch.keys())
            #     if i == 1:  # This will print the first batch
            #         break

            # print()
            # print('Fine-tuning Dataloader')
            # for i, batch in enumerate(task_dataloader):
            #     # print(batch)
            #     if i == 1:  # This will print the first batch
            #         break
            # print()

            print('')
            print('STARTING REALIGNMENT')
            print()

            freeze_model_using_strategy(strategy, model_name, model)

            log_layer_status(model, model_name)

            realignment_complete = (
                realignment_ckpt is not None
                and ckpt_iter is not None
                and ckpt_iter >= realignment_steps_before - 1
            )
            if not realignment_complete:
                if enable_weighted_sampling:
                    if weighted_sampling_method == "ucb":
                        logging.info("Performing UCB weighted sampling realignment.")
                        training_state = ucb_weighted_selection_realignment(
                            model=model,
                            tokenizer=tokenizer,
                            realignment_optimizer=before_optimizer,
                            alignment_datasets=realignment_dataset,
                            noaligner=noaligner,
                            ### UCB specific
                            ucb_exploration_coef=ucb_exploration_coef,
                            ### Shared weighted sampling
                            inner_batches_before_outer=inner_batches_before_outer,
                            lbsmoothing_eps=lbsmoothing_eps,
                            noise_mixing_strat=noise_mixing_strat,
                            softmax_temp=softmax_temp,
                            ###
                            realignment_steps=realignment_steps_before,
                            start_iter=ckpt_iter if ckpt_iter is not None else 0,
                            extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
                            realignment_coef=realignment_coef,
                            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
                            batch_size=realignment_batch_size,
                            logging_steps=logging_steps,
                            result_store=result_store,
                            training_state=training_state,
                            checkpoint_path=checkpoint_path,
                            checkpoint_prefix_name=realignment_checkpoint_prefix_name,
                            use_adapter=use_adapter,
                            adapter_approach=adapter_approach,
                            adapter_config=adapter_config,
                        )
                    else:
                        if decouple_meta_and_model_updates not in (
                            "joint", "decoupled", "decoupled_closed"
                        ):
                            raise ValueError(
                                f"decouple_meta_and_model_updates must be one of "
                                f"'joint', 'decoupled', 'decoupled_closed', got "
                                f"{decouple_meta_and_model_updates!r}"
                            )
                        if decouple_meta_and_model_updates == "joint":
                            _weighted_realignment_fn = weighted_selection_realignment
                            extra_kwargs = {}
                        else:
                            _weighted_realignment_fn = (
                                decoupled_weighted_selection_realignment
                            )
                            extra_kwargs = {
                                "closed_form_meta": decouple_meta_and_model_updates
                                == "decoupled_closed"
                            }
                            if static_meta_weights_jsonl is not None:
                                extra_kwargs["static_meta_weights_jsonl"] = static_meta_weights_jsonl
                        logging.info(
                            f"Performing {decouple_meta_and_model_updates} "
                            "weighted sampling realignment."
                        )
                        training_state = _weighted_realignment_fn(
                            model=model,
                            tokenizer=tokenizer,
                            realignment_optimizer=before_optimizer,
                            alignment_datasets=realignment_dataset,
                            noaligner=noaligner,
                            ### Weighted sampling
                            meta_loss_type=meta_loss_type,
                            meta_learning_rate=meta_learning_rate,
                            inner_batches_before_outer=inner_batches_before_outer,
                            lbsmoothing_eps=lbsmoothing_eps,
                            with_regularization=with_regularization,
                            lambda_entropy=lambda_entropy,
                            noise_mixing_strat=noise_mixing_strat,
                            softmax_temp=softmax_temp,
                            ###
                            realignment_steps=realignment_steps_before,
                            start_iter=ckpt_iter if ckpt_iter is not None else 0,
                            extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
                            realignment_coef=realignment_coef,
                            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
                            batch_size=realignment_batch_size,
                            logging_steps=logging_steps,
                            result_store=result_store,
                            training_state=training_state,
                            checkpoint_path=checkpoint_path,
                            checkpoint_prefix_name=realignment_checkpoint_prefix_name,
                            use_adapter=use_adapter,
                            adapter_approach=adapter_approach,
                            adapter_config=adapter_config,
                            data_collator=data_collator if decouple_meta_and_model_updates == "decoupled_closed" else None,
                            **extra_kwargs,
                        )
                else: 
                    training_state = epoch_loop(
                        model,
                        before_optimizer,
                        task_dataloader=None,
                        realignment_dataloader=realignment_dataloader,
                        task_accumulation_steps=1,
                        logging_steps=logging_steps,
                        log_in_wandb=log_in_wandb,
                        result_store=result_store,
                        nb_iter=realignment_steps_before,
                        start_iter=ckpt_iter if ckpt_iter is not None else 0,
                        extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
                        realignment_step_callbacks=realignment_step_callbacks,
                        training_state=training_state,
                        log_first_sample=True,
                        realignment_steps_by_finetuning=realignment_steps_by_finetuning,
                        model_name=model_name,
                        strategy=strategy,
                        checkpoint_path=checkpoint_path,
                        checkpoint_prefix_name=realignment_checkpoint_prefix_name,
                        use_adapter=use_adapter,
                        adapter_approach=adapter_approach,
                        adapter_config=adapter_config,
                    )

                res = training_state.log_state()
                if log_in_wandb:
                    wandb.log(res)
                if result_store:
                    result_store.log(res)

            if cache_path is not None:
                if isinstance(model, SimplifiedModelForRealignment):
                    raise Exception(f"SimplifiedModelForRealignment is not compatible with caching")
                logging.info(f"Saving realigned model: {model_hash}")
                model.save_pretrained(cache_path)

                with open(os.path.join(cache_path, "info.txt"), "w") as f:
                    f.write(string_to_hash + "\n")

                with open(training_state_path, "w") as f:
                    json.dump(dataclasses.asdict(training_state), f)

                with open(info_path, "w") as f:
                    f.write(hash_args + "\n")

            print('')
            print('DONE REALIGNMENT')
            print()

        unfreeze_model_with_strategy(strategy, model_name, model)   
        
        if use_adapter and adapter_approach in ["separate", "realign_only"]:
            if isinstance(model, SimplifiedModelForRealignment):
                model.model = model.model.merge_and_unload(adapter_names=["main_adapter"])
            else:
                model = model.merge_and_unload(adapter_names=["main_adapter"])
            if adapter_approach == "realign_only":
                for name, param in model.named_parameters():
                    if always_frozen_params and name in always_frozen_params:
                        continue
                    param.requires_grad = True
            
    if use_adapter and adapter_approach == "separate":
        # from peft import LoraConfig, PeftModel
        # model_to_adapt = model.model if isinstance(model, SimplifiedModelForRealignment) else model
        # adapter_config = LoraConfig(
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # Weirdly, peft and transformers library do not have the same method signature for add_adapter
        if isinstance(model_to_adapt, PeftModel):
            model_to_adapt.add_adapter("task", adapter_config)
        else:
            model_to_adapt.add_adapter(adapter_config, adapter_name="task")
        model_to_adapt.set_adapter("task")
    
    if "clirmatrix" in task_name: 
        return # Finetuning step will be done in the controlled_realignment file

    optimizer = get_optimizer()
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * len(task_dataloader) * n_epochs),
        num_training_steps=len(task_dataloader) * n_epochs,
    )

    for callback in epoch_callbacks:
        callback(model)

    print()
    print('STARTING FINETUNING')
    print()


    freeze_model_using_strategy(strategy, model_name, model)

    realignment_optimizer = None
    realignment_scheduler = None
    realignment_ignore_parameters = []
    if strategy.startswith("during_partial_freeze"):
        if isinstance(model, SimplifiedModelForRealignment):
            raise Exception(f"SimplifiedModelForRealignment is not compatible with strategy {strategy}")


        if "roberta" in model_name:
            n_layers = len(model.roberta.encoder.layer)
            encoder_prefix = "roberta.encoder.layer"
            embedding_prefix = "roberta.embeddings"
        elif "distilbert" in model_name:
            n_layers = len(model.distilbert.transformer.layer)
            encoder_prefix = "distilbert.transformer.layer"
            embedding_prefix = "distilbert.embeddings"
        elif model_name.startswith("bert"):
            n_layers = len(model.bert.encoder.layer)
            encoder_prefix = "bert.encoder.layer"
            embedding_prefix = "bert.embeddings"
        else:
            raise NotImplementedError(f"during_partial_freeze_* strategies are not implemented for model {model}")
        
        if strategy.endswith("front"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2)]
        elif strategy.endswith("back"):
            prefixes_to_ignore = [f"{encoder_prefix}.{i}" for i in range(n_layers // 2, n_layers)]
        elif strategy.endswith("none"):
            prefixes_to_ignore = []
        elif re.match(r"during_partial_freeze_[0-9]+_[0-9]+", strategy):
            *_, first_layer, last_layer = strategy.split("_")
            first_layer = int(first_layer)
            last_layer = int(last_layer)
            prefixes_to_ignore = []
            if first_layer == 0:
                prefixes_to_ignore.append(embedding_prefix)
                first_layer = 1
            for i in range(first_layer - 1, last_layer - 1):
                prefixes_to_ignore.append(f"{encoder_prefix}.{i}")
        else:
            raise NotImplementedError(f"Unrecognized strategy {strategy}")
        
        for name, param in model.named_parameters():
            if any(map(lambda x: name.startswith(x), prefixes_to_ignore)):
                realignment_ignore_parameters.append(name)
        
        if not strategy.endswith("none"):
            assert len(realignment_ignore_parameters) > 0

    log_layer_status(model, model_name)
    
    if not isinstance(evaluation_datasets, list) and evaluation_datasets is not None:
        evaluation_datasets = [evaluation_datasets]

    finetuning_ckpt, ft_prefix_name, checkpoint_epoch = load_ckpt_from_path(
        ckpt_type="finetuning",
        checkpoint_path=checkpoint_path,
        seed=seed,
        model_slug=model_slug,
        task_name=task_name,
        model=model,
        use_adapter=use_adapter,
        adapter_approach=adapter_approach,
        adapter_config=adapter_config,
        optimizer=optimizer,
        scheduler=scheduler,
        realignment_steps_before=nb_realignment_steps_before,
        )

    for i in range(n_epochs):
        if checkpoint_epoch is not None and i <= checkpoint_epoch:
            logging.info(f"Skipping epoch {i} as it is already completed in checkpoint.")
            continue

        if i == n_epochs - 1:
            logging.info(f"Saving checkpoint for epoch {i}")
            ft_checkpoint_path = checkpoint_path
            if realignment_steps_before: 
                ft_prefix_name = f"finetuning_{model_slug}_{task_name}_seed_{seed}_reiter_{realignment_steps_before - 1}_epoch_{i}"
            else:
                ft_prefix_name = f"finetuning_{model_slug}_{task_name}_seed_{seed}_epoch_{i}"
        else:
            ft_checkpoint_path = ft_prefix_name = None
        
        training_state = epoch_loop(
            model,
            optimizer,
            scheduler=scheduler,
            task_dataloader=task_dataloader,
            realignment_dataloader=realignment_dataloader
            if "during" in strategy or strategy == "staged"
            else None,
            realignment_optimizer=realignment_optimizer,
            realignment_scheduler=realignment_scheduler,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            realignment_coef=realignment_coef
            if realignment_coef_scheduler is None
            else realignment_coef_scheduler(i),
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            log_first_sample=i == 0,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
            extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
            separate_backward=strategy == "during_separate_backward" or bool(realignment_ignore_parameters),
            realignment_ignore_parameters=realignment_ignore_parameters,
            checkpoint_path=ft_checkpoint_path,
            checkpoint_prefix_name=ft_prefix_name,
            use_adapter=use_adapter,
            adapter_approach=adapter_approach,
            adapter_config=adapter_config,
        )
        for callback in epoch_callbacks:
            callback(model)

        res = training_state.log_state()
        if realignment_ckpt:
            res["realignment_steps"] = realignment_ckpt['nb_iter']
            res["realignment_loss"] = realignment_ckpt['loss']
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

        if evaluation_datasets is not None:
            frozen_params = freeze_all(model)
            res = evaluate_several_token_classification(
                tokenizer,
                model,
                evaluation_datasets,
                batch_size=task_batch_size,
                prefixes=evaluation_prefixes,
                overall_prefix="eval",
                metric_fn=metric_fn,
                collator=data_collator,
                label_key=label_key,
            )
            unfreeze_all(model, frozen_params)
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)
        if same_language_evaluation_dataset is not None:
            frozen_params = freeze_all(model)
            res = evaluate_token_classification(
                model, same_language_evaluation_dataloader, prefix="eval_same", metric_fn=metric_fn
            )
            unfreeze_all(model, frozen_params)
            logging.info(res)
            if log_in_wandb:
                wandb.log(res)
            if result_store:
                result_store.log(res)

    print()
    print('DONE FINETUNING')
    print()

    if strategy == "after":
        after_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        training_state = epoch_loop(
            model,
            after_optimizer,
            task_dataloader=None,
            realignment_dataloader=realignment_dataloader,
            task_accumulation_steps=accumulation_steps,
            logging_steps=logging_steps,
            log_in_wandb=log_in_wandb,
            result_store=result_store,
            nb_iter=(
                len(task_dataloader) * n_epochs
                if nb_realignment_steps_before is None
                else nb_realignment_steps_before * accumulation_steps
            ),
            extra_realignment_steps_checkpoints=extra_realignment_steps_checkpoints,
            realignment_step_callbacks=realignment_step_callbacks,
            training_state=training_state,
            realignment_steps_by_finetuning=realignment_steps_by_finetuning,
            use_adapter=use_adapter,
            adapter_approach=adapter_approach,
            adapter_config=adapter_config,
        )
        res = training_state.log_state()
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
        for callback in epoch_callbacks:
            callback(model)

    if evaluation_datasets is not None:
        frozen_params = freeze_all(model)
        res = evaluate_several_token_classification(
            tokenizer,
            model,
            evaluation_datasets,
            batch_size=task_batch_size,
            prefixes=evaluation_prefixes,
            overall_prefix=f"{final_prefix}_eval",
            metric_fn=metric_fn,
            collator=data_collator,
            label_key=label_key,
        )
        unfreeze_all(model, frozen_params)
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)
    if same_language_evaluation_dataset is not None:
        frozen_params = freeze_all(model)
        res = evaluate_token_classification(
            model,
            same_language_evaluation_dataloader,
            prefix=f"{final_prefix}_eval_same",
            metric_fn=metric_fn,
        )
        unfreeze_all(model, frozen_params)
        logging.info(res)
        if log_in_wandb:
            wandb.log(res)
        if result_store:
            result_store.log(res)

    if return_model_hash and use_caching:
        return model_hash
