from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import logging
import os

from multilingual_eval.datasets.data_utils import TorchCompatibleIterableDataset
from multilingual_eval.datasets.realignment_dataset import (
    RealignmentAndOtherCollator, RealignmentCollator
)
from multilingual_eval.datasets.realignment_task import (
    get_realignment_dataset_for_one_pair,
)

import torch
from torch.utils.data import DataLoader

def _validate_noise_config(noise_mixing_strat, batch_size, num_batches):
        """
        Parses and validates the noise mixing strategy.
        Format: {strat}_{noise_type}_{amount} (e.g., 'examples_uniform_5')
        """
        # 1. Parsing with detailed error reporting
        try:
            # Split into [strat, noise_type, amount]
            parts = noise_mixing_strat.split("_")
            if len(parts) < 3:
                raise ValueError
            
            # amount is always the last element
            amount = int(parts[-1])
            # noise_type is the second to last
            noise_type = parts[-2]
            # strat is everything before that (joined back if it had underscores)
            strat = "_".join(parts[:-2])
            
        except (ValueError, IndexError):
            raise ValueError(
                f"Incorrect syntax: '{noise_mixing_strat}'. "
                "Expected format: {strat}_{noise_type}_{amount} (e.g., 'examples_reverse_10')"
            )

        # 2. Strategy & Type Validation
        valid_strats = ["examples", "batches"]
        valid_noises = ["uniform", "reverse"]

        if strat not in valid_strats:
            raise ValueError(f"Unrecognized strategy '{strat}'. Must be one of {valid_strats}.")
        
        if noise_type not in valid_noises:
            raise ValueError(f"Unrecognized noise_type '{noise_type}'. Must be one of {valid_noises}.")

        # 3. Logical Constraints
        if amount <= 0:
            raise ValueError(f"Amount of noise must be > 0. Got: {amount}")

        if strat == "examples" and amount >= batch_size:
            raise ValueError(
                f"For 'examples' strategy, amount ({amount}) must be < batch_size ({batch_size})."
            )

        if strat == "batches" and amount >= num_batches:
            raise ValueError(
                f"For 'batches' strategy, amount ({amount}) must be < number of inner loop batches ({num_batches})."
            )

        return strat, noise_type, amount

def _generate_lang_ids(probs, batch_size, num_batches, noise_mixing_strat=None, device="cuda"):
        # 1. Base case: No noise
        if not noise_mixing_strat:
            num_samples = batch_size * num_batches
            return torch.multinomial(probs, num_samples=num_samples, replacement=True).to(device)

        # 2. Parse config
        strat, n_type, n_amount = _validate_noise_config(noise_mixing_strat, batch_size, num_batches)
        
        # Pre-calculate inverse probs if needed for "reverse"
        if n_type == "reverse":
            inverse_probs = (1.0 - probs).clamp(min=1e-6) # Avoid zero probs
            inverse_probs /= inverse_probs.sum()

        # 3. Handle "examples" strategy (Noise inside every batch)
        if strat == "examples":
            # Sample clean IDs for all batches at once: (num_batches, clean_per_batch)
            clean_samples = torch.multinomial(probs, num_samples=num_batches * (batch_size - n_amount), replacement=True)
            clean_samples = clean_samples.view(num_batches, batch_size - n_amount)

            # Sample noise IDs for all batches at once: (num_batches, noise_per_batch)
            if n_type == "uniform":
                noise_samples = torch.randint(0, len(probs), (num_batches, n_amount), device=device)
            else: # reverse
                noise_samples = torch.multinomial(inverse_probs, num_samples=num_batches * n_amount, replacement=True)
                noise_samples = noise_samples.view(num_batches, n_amount)

            # Concatenate and flatten: (num_batches, batch_size) -> (total_samples,)
            return torch.cat([clean_samples, noise_samples], dim=1).view(-1).to(device)

        # 4. Handle "batches" strategy (Entire batches are noise)
        elif strat == "batches":
            clean_count = num_batches - n_amount
            
            # All clean batches
            clean_samples = torch.multinomial(probs, num_samples=clean_count * batch_size, replacement=True)
            
            # All noisy batches
            if n_type == "uniform":
                noise_samples = torch.randint(0, len(probs), (n_amount * batch_size,), device=device)
            else: # reverse
                noise_samples = torch.multinomial(inverse_probs, num_samples=n_amount * batch_size, replacement=True)
                
            return torch.cat([clean_samples, noise_samples]).to(device)

def weighted_multilingual_batch(
    probs=None,                # tensor of shape (num_langs,)
    noise_mixing_strat=None,
    batch_size=32,
    num_batches=1,
    iterators=None,            # dict: lang -> iterator
    datasets=None,             # dict: lang -> IterableDataset
    tokenizer=None,
    noaligner=False,
    device="cuda"
):
    """
    Fetch a multilingual batch for inner (train) or outer (dev/meta) loop.

    - For train: sample according to `weights` over `selected_langs`.
    - For dev: sample uniformly (unless weighted dev is specified).
    noise_mixing_strat: Strategy to mix in noise (within or entirely batches) and the mixing amount.
    """

    lang_ids = _generate_lang_ids(probs, batch_size, num_batches, noise_mixing_strat, device)

    # Collect samples
    samples = []
    for lang_id in lang_ids:
        it = iterators[lang_id]

        try:
            samples.append(next(it))
        except StopIteration:
            logging.warning(f"Lang id {lang} reached end of dataset, restarting.")
            iterators[lang] = iter(datasets[lang])
            it = iterators[lang]
            samples.append(next(it))
            continue

    # Wrap as DataLoader
    return DataLoader(
        list(zip(samples, lang_ids)), # each element: (sample, lang_id)
        shuffle=False,
        batch_size=batch_size,
        collate_fn=RealignmentCollator(
            tokenizer,
            noaligner=noaligner,
        ),
    )