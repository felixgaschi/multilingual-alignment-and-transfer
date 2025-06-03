# Standard library imports
import os
import sys
import re
import logging
import argparse
import random

# Third-party imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Local application imports
sys.path.append(os.curdir)
from multilingual_eval.models.with_realignment_factory import AutoModelForSequenceClassificationWithRealignment

# Configure logging
from datetime import datetime

today_date = datetime.now().strftime("%Y-%m-%d")
file_name = os.path.basename(__file__).split('.')[0]
logging.basicConfig(filename=f"{file_name}_{today_date}.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

# Set random seed for reproducibility
random.seed(42)

def load_model(model_name="distilbert-base-multilingual-cased", device="cpu", checkpoint_file=None, strategy=''):    
    # Load model and tokenizer
    model_cache_dir = "/home/bumie304/scratch/nlp_project/cache/transformers"
    Model_Class = AutoModelForSequenceClassification if strategy == "baseline" else AutoModelForSequenceClassificationWithRealignment
    model = Model_Class.from_pretrained(model_name, cache_dir=model_cache_dir, local_files_only=True, num_labels=3).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, local_files_only=True)
    logging.info("Loaded pre-trained model")

    if checkpoint_file:
        try:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Checkpoint loaded {checkpoint_file}")
        except Exception as e:
            raise FileExistsError(f"Cannot load checkpoint: {checkpoint_file}.\nError: {e}")

    return model, tokenizer

def extract_model_embs(model_name, strategy_dir, strategy, device, langs, batch_size, num_sent, save_dir):
    logging.info(f"__________Processing model: {model_name}\nStrategy: {strategy}")
    layers = range(1, 7) if "distilbert" in model_name else range(1, 13)
    if strategy == "pretrain":
        for layer in tqdm(layers, desc="Layers", position=2):
            logging.info(f"____Processing layer: {layer}")
            extract_layer_embeds(model_name, device, langs, batch_size, layer, num_sent, save_dir)
    else:
        checkpoint_dir = os.path.join(strategy_dir, strategy, f"{model_name}__opus100")
        for checkpoint_file in tqdm(os.listdir(checkpoint_dir), desc="Checkpoints", position=2):
            info = extract_info(checkpoint_file)
            for layer in tqdm(layers, desc="Layers", position=3):
                logging.info(f"____Processing layer: {layer}")
                extract_layer_embeds(model_name, device, langs, batch_size, layer, strategy, info, num_sent, checkpoint=os.path.join(checkpoint_dir, checkpoint_file), save_dir=save_dir)

def extract_info(filename):
    """
    Extract step, seed, and, if applicable, epoch from the given filename.

    Expected filename format:
    - For finetuning: "finetuning_[model]_seed_<seed>_epoch_<epoch>_..."
    - For realignment: "realignment_[model]_seed_<seed>_..."
    
    Returns:
        A dictionary with keys: 'step', 'seed', and optionally 'epoch' if step is 'finetuning'.
    """
    pattern = r'^(?P<step>finetuning|realignment)_.+?_seed_(?P<seed>\d+)(?:_epoch_(?P<epoch>\d+))?'

    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Filename did not match the expected pattern.")
    
    # Extract mandatory fields.
    step = match.group("step")
    seed = int(match.group("seed"))
    result = {"step": step, "seed": seed}
    
    # For finetuning, we expect an epoch value.
    if step == "finetuning":
        epoch_str = match.group("epoch")
        if epoch_str is None:
            raise ValueError("Epoch value missing in filename for finetuning.")
        result["epoch"] = int(epoch_str)
    
    return result

# reference to https://github.com/kathyhaem/outliers
# paper link: https://aclanthology.org/2023.findings-acl.439/
def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
     # Expand the attention mask to match the shape of token embeddings
    attention_mask_expanded = attention_mask.unsqueeze(-1).float()

    # Compute the weighted sum of the embeddings, considering the attention mask
    weighted_sum = torch.sum(token_embeddings * attention_mask_expanded, dim=1)

    # Count the number of non-padding tokens (sum of the attention mask)
    non_padding_count = attention_mask_expanded.sum(dim=1)

    # Avoid division by zero by clamping the non-padding count to a minimum value
    return weighted_sum / torch.clamp(non_padding_count, min=1e-9)

def get_embeds(data, model, tokenizer, batch_size, layer, device):
    tgt_embeddings_layer = []
    attention_masks = []

    # Prepare the dataloader
    dataloader = DataLoader(data, batch_size=batch_size, drop_last=False)

    # Compute the max length of input sequences just once before the loop
    max_len = max(len(x) for x in tokenizer(data, padding=True, truncation=True)['input_ids'])

    for batch in dataloader:
        # Tokenize each batch
        tokens = tokenizer(batch, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt').to(device)

        # Collect attention masks
        attention_masks.append(tokens['attention_mask'].cpu())

        # Get model embeddings with no gradient tracking
        with torch.no_grad():
            output = model.forward(**tokens, output_hidden_states=True)
            layer_embeddings = output.hidden_states[layer].cpu()

            # Append embeddings of the current batch to the list
            tgt_embeddings_layer.append(layer_embeddings)

    # Concatenate all batches at once and move to CPU
    tgt_embeddings_layer = torch.cat(tgt_embeddings_layer, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Apply mean pooling
    return mean_pooling(tgt_embeddings_layer, attention_masks)

def extract_layer_embeds(model_name, device, langs, batch_size, layer, strategy, info, num_sent, checkpoint=None, save_dir=""):
    # Load the model and tokenizer once before processing languages
    model, tokenizer = load_model(model_name, device, checkpoint, strategy)
    dataset_dir = os.path.join(os.getcwd(), "data", "opus100")

    # Iterate over each language in 'langs'
    for lang in langs:
        logging.info(f"Currently processing {lang}...")

        # Directory for saving embeddings, creating it if it doesn't exist
        lang_dir = f"{save_dir}/{strategy}/{model_name}/{info['step']}/layer_{layer}/{lang}"
        os.makedirs(lang_dir, exist_ok=True)

        if info['step'] == "finetuning":
            embed_info = f"epoch_{info['epoch']}_seed_{info['seed']}"
        else:
            embed_info = f"seed_{info['seed']}"

        tar_save_path = os.path.join(lang_dir, f'{lang}_{embed_info}.pt')
        en_save_path = os.path.join(lang_dir, f'en_{embed_info}.pt')
        pair_name = '-'.join(sorted(['en', f'{lang}']))
        if not os.path.exists(tar_save_path):
            target_sents =  open(f"{dataset_dir}/{pair_name}/opus.{pair_name}-train.{lang}", "r").readlines()
            target_sents = random.sample(target_sents, num_sent)
            target_embeddings = get_embeds(target_sents, model, tokenizer, batch_size, layer, device)
            torch.save(target_embeddings, tar_save_path)
        else:
            logging.info(f"{tar_save_path} already exist, skipping...")

        if not os.path.exists(en_save_path):
            en_sents = open(f"{dataset_dir}/{pair_name}/opus.{pair_name}-train.en", "r").readlines()
            en_sents = random.sample(en_sents, num_sent)
            en_embeddings = get_embeds(en_sents, model, tokenizer, batch_size, layer, device)
            torch.save(en_embeddings, en_save_path)
        else:
            logging.info(f"{en_save_path} already exist, skipping...")

        logging.info(f"Finished saving embeddings for {lang} in model {model_name}.")

def parse_args():
    # Determine the default device based on torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default values for other parameters
    default_batch_size = 32
    default_model_list = [
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
    ]
    default_langs = "bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh".split()
    default_num_sent = 1000
    default_save_dir = "data/embs/"
    default_strategy_dir = "data/results/"
    default_strategy = ['pretrain']
    
    parser = argparse.ArgumentParser(
        description="Parse configuration arguments for the experiment."
    )
    
    parser.add_argument(
        "--strategy_dir",
        type=str,
        default=default_strategy_dir,
        help="Directory containing strategy checkpoints (default: %(default)s)"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=default_strategy,
        help="List of strategy checkpoints %(default)s"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to run on (default: cuda if available, else cpu)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="Batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--model-list",
        nargs="+",
        default=default_model_list,
        help="List of models (default: %(default)s)"
    )

    parser.add_argument(
        "--langs",
        nargs="+",
        default=default_langs,
        help="List of language codes (default: %(default)s)"
    )
    
    parser.add_argument(
        "--num-sent",
        type=int,
        default=default_num_sent,
        help="Number of sentences (default: %(default)s)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=default_save_dir,
        help="Directory to save outputs (default: %(default)s)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    logging.info("Parsed arguments:")
    logging.info(f"Device: {args.device}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Model list: {args.model_list}")
    logging.info(f"Languages: {args.langs}")
    logging.info(f"Number of sentences: {args.num_sent}")
    logging.info(f"Save directory: {args.save_dir}")
    logging.info(f"Strategy directory: {args.strategy_dir}")
    logging.info(f"Strategy: {args.strategies}")

    # Check if the strategy directory exists and is not empty
    if not (os.path.exists(args.strategy_dir) and os.listdir(args.strategy_dir)):
        raise FileNotFoundError(f"Strategy directory '{args.strategy_dir}' does not exist or is empty.")

    # Get the existing strategy folders in the directory
    available_strategies = os.listdir(args.strategy_dir)

    for strategy in tqdm(args.strategies, desc="Strategies", position=0):
        # Check if requested strategies exist
        if strategy not in available_strategies:
            logging.warning(f"Strategy '{strategy}' not found in '{args.strategy_dir}'. Skipping...")
            continue
        for model_name in tqdm(args.model_list, desc="Models", position=1):
            extract_model_embs(model_name, args.strategy_dir, strategy, args.device, args.langs, args.batch_size, args.num_sent, args.save_dir)