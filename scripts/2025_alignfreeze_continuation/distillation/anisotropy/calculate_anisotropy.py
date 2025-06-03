import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import argparse
import logging

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

def cos_contrib(emb1, emb2):
    """ Adopted as-is from https://github.com/wtimkey/rogue-dimensions/blob/main/replication.ipynb
    Code from that repo is licensed under Apache 2.0: https://github.com/wtimkey/rogue-dimensions/blob/main/LICENSE"""
    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return np.array(numerator_terms / denom)

def calculate_anisotropy(langs, seed, epoch, verbose, num_pair=1000, embs_dir=""):
    avg_anisotropy = 0
    mean_contribs = []
    for lang in langs:
        # print(f"Current language: {lang}")
        """ Adapted cosine contribution calculation from
        https://github.com/wtimkey/rogue-dimensions/blob/main/replication.ipynb
        Changes: We add a version for parallel data and code for handling different datasets; we only calculate
        contributions for one layer at a time but consider multiple languages, which we also average in the end.
        We calculate up to the top ten contributing dimensions.
        """
        layer_cosine_contribs = []
        file_tail = f"epoch_{epoch}_seed_{seed}" if epoch != -100 else f"seed_{seed}"
        target_embs = torch.load(f'{embs_dir}/{lang}/{lang}_{file_tail}.pt')
        eng_embs = torch.load(f'{embs_dir}/{lang}/en_{file_tail}.pt')
        num_sents = target_embs.shape[0]
        # randomly sample embedding pairs
        random_pairs = [random.sample(range(num_sents), 2) for _ in range(num_pair)]

        for pair in random_pairs:
            emb1, emb2 = target_embs[pair[0]], eng_embs[pair[1]]
            layer_cosine_contribs.append(cos_contrib(emb1, emb2))
        layer_cosine_contribs = np.stack(layer_cosine_contribs)
        layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)

        aniso = layer_cosine_contribs_mean.sum()
        avg_anisotropy += aniso
        mean_contribs.append(layer_cosine_contribs_mean)

        top_dims = np.argsort(layer_cosine_contribs_mean)[-10:]
        top_dims = np.flip(top_dims)

        if verbose:
            print(f"### {lang} ###")
            print(f"Top 10 dims: {top_dims}")
            print(f"Estimated anisotropy: {aniso}")
        
            print("Contributions to expected cosine sim between random embeddings:")
            for i in range(10):
                d = top_dims[i]
                print(d, layer_cosine_contribs_mean[d])

    print()
    avg_anisotropy = avg_anisotropy / len(langs)
    print(f"Average Anisotropy: {avg_anisotropy}")

    mean_contribs = np.stack(mean_contribs).mean(axis=0)
    top_dims = np.argsort(mean_contribs)[-5:]
    top_dims = np.flip(top_dims)
    print(f"Top 5 dims: {top_dims}")
    print("Mean cosine contributions:")

    top_dim_and_mean_contrib = []
    for i in range(5):
        d = top_dims[i]
        mean_contrib = round(mean_contribs[d], 3)
        print(d, mean_contrib)
        top_dim_and_mean_contrib.append((d, mean_contrib))
    return avg_anisotropy, top_dim_and_mean_contrib

def process_model(model_name, args, rows, strategy):
    logging.info(f"___Model: {model_name}")
    if "distilbert" in model_name:
        layer_range = range(1, 7)
    else: 
        layer_range = range(1, 13)
    model_dir = f"{args.embs_dir}/{strategy}/{model_name}"
    steps = os.listdir(f"{model_dir}") if args.step == "all" else [args.step]
    for step in tqdm(steps, desc="Steps", position=2):
        logging.info(f"______Step: {step}")
        for layer in layer_range:
            process_layer(layer, args, step, rows, strategy)

def process_layer(layer, args, step, rows, strategy):
    logging.info(f"_________Layer: {layer}")
    current_dir = os.path.join(os.getcwd(), f"{args.embs_dir}/{strategy}/{model_name}/{step}/layer_{layer}")
    for seed in args.seed_list:
        logging.info(f"____________Seed: {seed}")
        epoch_list = args.epoch if step == "finetuning" else [-100]
        for epoch in epoch_list:
            logging.info(f"_______________Epoch: {epoch}")
            avg_anisotropy, top_dim_and_mean_contrib = calculate_anisotropy(langs=args.langs, seed=seed, epoch=epoch,
                                                                            verbose=False, num_pair=args.num_pair,
                                                                            embs_dir=current_dir)
            rows.append({
                            "model": model_name,
                            "layer": layer,
                            "avg_anisotropy_xlang": avg_anisotropy,
                            "top_3_dims_and_contribs": top_dim_and_mean_contrib,
                            "step": step,
                            "seed": seed,
                            "epoch": epoch,
                        })

def parse_args():
    # Default values
    default_langs = "bg cs de es lv af ar ca da el fa fi fr he hi hu it ko lt no pl pt ro ru sk sl sv ta th tr uk vi".split()
    default_model_list = [
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
    ]
    default_embs_dir = "data/embs"
    default_strategies = ["baseline"]
    default_num_pair = 1000
    default_seed_list = [31, 42, 66]
    default_epoch = [0, 1]
    default_step = "all"

    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for the anisotropy calculation script."
    )

    # Adding arguments with default values
    parser.add_argument(
        "--langs",
        nargs="+",
        default=default_langs,
        help="List of language codes. (default: %(default)s)"
    )
    parser.add_argument(
        "--model-list",
        nargs="+",
        default=default_model_list,
        help="List of model names. (default: %(default)s)"
    )
    parser.add_argument(
        "--embs-dir",
        type=str,
        default=default_embs_dir,
        help="Directory to load embeddings. (default: %(default)s)"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=default_strategies,
        help="List of strategies to use. (default: %(default)s)"
    )
    parser.add_argument(
        "--num-pair",
        type=int,
        default=default_num_pair,
        help="Number of pairs (default: %(default)s)"
    )

    parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=default_seed_list,
        help="List of seed values (default: %(default)s)"
    )
    
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        default=default_epoch,
        help="List of epoch values (default: %(default)s)"
    )
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "realignment", "finetuning"],
        default=default_step,
        help="Step to perform: 'realignment' or 'finetuning', or 'all' for all steps (default: %(default)s)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Log the parsed arguments
    logging.info("Parsed arguments:")
    logging.info("Languages: %s", args.langs)
    logging.info("Model list: %s", args.model_list)
    logging.info("Embeddings directory: %s", args.embs_dir)
    logging.info("Strategies: %s", args.strategies)
    logging.info("Seed list: %s", args.seed_list)
    logging.info("Epoch: %s", args.epoch)
    logging.info("Step: %s", args.step)
    
    # For each strategy, process the models and save the results.
    for strategy in tqdm(args.strategies, desc="Strategies", position=0):
        logging.info("Processing strategy: %s", strategy)

        # Build the saving directory path
        saving_dir = f"scripts/2025_alignfreeze_continuation/distillation/anisotropy_results/{strategy}"
        os.makedirs(saving_dir, exist_ok=True)

        # Accumulate rows in a list to later convert into a DataFrame.
        rows = []
        for model_name in tqdm(args.model_list, desc="Models", position=1):
            process_model(model_name, args, rows, strategy)

        # Create the DataFrame.
        result_dataframe = pd.DataFrame(
            rows, 
            columns=["model", "layer", "avg_anisotropy_xlang", "top_3_dims_and_contribs", "step", "seed", "epoch"]
        )

        file_name = f"{saving_dir}/anisotropy_{strategy}_{args.num_pair}_pairs.csv"
        result_dataframe.to_csv(file_name, index=False, mode='a', header=not os.path.exists(file_name))
        logging.info("Results saved to %s", file_name)