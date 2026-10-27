# Online Language Adaptive Sampling for Better Distributed Cross-lingual Gains

Code for the paper: [Online Language Adaptive Sampling for Better Distributed Cross-lingual Gains](https://arxiv.org/abs/2609.14969). Quang Phuoc Nguyen, Félix Gaschi, David Anugraha, Santiago Martínez Novoa, En-Shiun Annie Lee. (EMNLP 2026)

This directory contains the scripts for reproducing the experiments from the above-mentioned paper. The source code relies on the `multilingual_eval` library at the root of this repository.

Instead of sampling parallel sentences uniformly at random, the method gives each realignment language a trainable sampling probability, so that languages contributing more to the realignment loss get sampled more often. Two variants are provided: a gradient-based one (meta-learning over the sampling weights, with an inner/outer optimization loop) and a UCB-based one.

## Requirements

This was tested with Python 3.12 using [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies are specified in `pyproject.toml` at the root of the repository.

To install uv and set up the environment:

```bash
pip install uv
uv sync
```

LoRA fine-tuning (used for the decoder-based models) additionally requires `peft`, which is part of the dependencies.

## Data

As in [scripts/2025_aacl](../2025_aacl/README.md), the experiments use parallel corpora from two sources:

- [OPUS-100](https://opus.nlpl.eu/OPUS-100.php) for high- and medium-resource languages
- [NLLB-200](https://huggingface.co/datasets/allenai/nllb) for low-resource languages (primarily African languages)

To download the full dataset:

```bash
bash scripts/2026_emnlp/download_resources.sh <DATA_DIR>
```

This takes a while (more than 12 hours) and a lot of storage, as it retrieves and, for Chinese, segments the parallel data used for realignment.

## How to use?

The main entry point is `controlled_realignment.py`, driven by `final_run_script.sh`:

```bash
bash scripts/2026_emnlp/final_run_script.sh \
    <STRATEGY> \
    <SELECTION_STRAT> \
    <MODEL> \
    <TASK> \
    <SEED> \
    16000 \
    128 \
    <ADDITIONAL_ARGS>
```

where:

- `<STRATEGY>` is the realignment strategy, one of:
  - `baseline`: fine-tuning only (no realignment)
  - `before_noaligner`: realignment using sentence-level averaging (no word aligner required)

  The codebase supports other aligners, but they are not relevant to this paper and the script does not expose them.
- `<SELECTION_STRAT>` is the language selection strategy, one of:
  - `xt_afri`: all 65 languages (XTREME-R + African), which is the pool the adaptive sampling runs on
  - `xt_only` / `afri_only`: the XTREME-R or African subsets only
  - `most_uriel_en_40` / `least_uriel_en_40`: the 40 most (resp. least) URIEL-featurally distant languages from English, from [Nguyen et al., 2025](../2025_aacl/README.md)
  - `baseline`: fine-tuning only
- `<MODEL>` is any Huggingface model identifier. The results in this directory cover `xlm-roberta-large`, `google-bert/bert-base-multilingual-cased`, `distilbert-base-multilingual-cased`, `microsoft/mdeberta-v3-base`, `jhu-clsp/mmBERT-base`, `google/mt5-small`, `google/mt5-large`, `google/gemma-2-2b`, `google/gemma-2-9b`, `meta-llama/Llama-3.2-3B` and `meta-llama/Llama-3.1-8B`.
- `<TASK>` is one of `xnli`, `wikiann`, `xtreme_r.udpos`, `xquad`, `sib200`, `xlsum`, or a `clirmatrix` variant.
- `<SEED>` is an integer, or a string containing several integers.
- The two numeric arguments are the maximum number of realignment steps and the realignment batch size.
- `<ADDITIONAL_ARGS>` are extra arguments forwarded to the training script. **They must be wrapped in a single string so they are passed as one argument.** For the two sampling variants:
  - gradient-based: `"--enable_weighted_sampling --weighted_sampling_method meta_learning --inner_batches_before_outer 10 --meta_learning_rate 1e-3"`
  - UCB-based: `"--enable_weighted_sampling --weighted_sampling_method ucb --ucb_exploration_coef 0.1"`

  Add `--decouple_meta_and_model_updates decoupled` (or `decoupled_closed` for the closed-form variant) to the gradient-based arguments to decouple the meta-weight updates from the model updates; this is the setting behind the `__decouple__` result directories.

### Toy reproduction

Toy scripts reproduce the pipeline on a smaller subset of languages (6 OPUS-100 + 4 NLLB) and less data, which runs relatively quickly for testing purposes. They still require more than 12GB of storage.

```bash
bash scripts/2026_emnlp/download_resources_toy.sh <DATA_DIR>
bash scripts/2026_emnlp/final_run_script_toy.sh \
    <STRATEGY> \
    <SELECTION_STRAT> \
    <MODEL> \
    <TASK> \
    <SEED> \
    16000 \
    128 \
    <ADDITIONAL_ARGS>
```

The toy run script takes the same arguments as the full one and evaluates each task only on the languages for which the toy download provides data.

## Results

`scripts/2026_emnlp/results` holds the results reported in the paper as CSV files, with one directory per method. Each directory is named after the realignment setting it was run with, for instance `ft_only` for the fine-tuning baseline, `xt_afri` for uniform realignment over the 65 languages, or `xt_afri__weighted__decouple__innerb10__metalr1e-3` for the decoupled gradient-based variant. `final_run_script.sh` derives that name from its own arguments, so a new run lands in the directory of its method, next to the runs released here.

Within a directory, each file is named `<model>__<dataset>__<strategy>__<task>__<realignment steps>__<realignment batch size>.csv` and holds one row per seed.

## What is specific to this paper

The reusable parts live in `multilingual_eval` at the root of the repository. The following were added or extended for this paper:

| Path | Description |
|------|-------------|
| `multilingual_eval/datasets/weighted_data_utils.py` | Weighted sampling of the parallel data |
| `multilingual_eval/training/decoupled_weighted_selection.py` | Decoupled meta-weight learning for language selection |
| `multilingual_eval/training/epoch_loop.py` | `weighted_selection_realignment` and `ucb_weighted_selection_realignment`, and checkpoint resumption across meta-learning iterations |
| `multilingual_eval/training/training_loops.py` | Integration of weighted sampling and of the decoupled weight updates |
| `multilingual_eval/models/realignment_loss.py` | Closed-form solution variant |
| `multilingual_eval/models/simplified.py`, `models/stwrapper.py` | Simplified model variants for the ablations, and a sentence-transformer wrapper for CLIR |
| `multilingual_eval/information_retrieval/` | Cross-lingual retrieval: MS MARCO training, CLIRMatrix evaluation, shared data utilities |
| `multilingual_eval/datasets/xlsum.py`, `datasets/topic_classification.py` | XL-Sum and SIB-200 support |
