# Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages

Code for the paper: Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages. Quang Phuoc Nguyen\*, David Anugraha\*, Felix Gaschi\*, Jun Bin Cheng, En-Shiun Annie Lee. (IJCNLP-AACL 2025)

This directory contains the scripts for reproducing the experiments from the above-mentioned paper. The source code relies on the `multilingual_eval` library at the root of this repository.

## Requirements

This was tested with Python 3.12 using [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies are specified in `pyproject.toml` at the root of the repository.

To install uv and set up the environment:

```bash
pip install uv
uv sync
```

## Data

The experiments use parallel corpora from two sources:
- [OPUS-100](https://opus.nlpl.eu/OPUS-100.php) for high- and medium-resource languages
- [NLLB-200](https://huggingface.co/datasets/allenai/nllb) for low-resource languages (primarily African languages)

The script `scripts/2025_aacl/download_resources.sh` downloads all necessary data and prepares it for training:

```bash
bash scripts/2025_aacl/download_resources.sh <DATA_DIR>
```

where `<DATA_DIR>` is the directory where the datasets will be stored. This may take a while depending on your connection.

**Note:** If you plan to use the FastAlign-based realignment strategy (`before_fastalign`), you will also need [FastAlign](https://github.com/clab/fast_align) installed and available in your `PATH`. The download script prepares FastAlign-compatible tokenized files for OPUS-100 languages. NLLB-200 languages are only supported with the `noaligner` strategy.

## How to use?

To run a realignment experiment, use `scripts/2025_aacl/run_langs_selection.sh`:

```bash
bash scripts/2025_aacl/run_langs_selection.sh <DATA_DIR> <DATASET> <STRATEGY> <SEED> <SELECTION_STRAT> <TASK>
```

where:
- `<DATA_DIR>` is the path to the data directory used in `download_resources.sh`
- `<DATASET>` is the parallel dataset subdirectory to use (e.g., `mix_opus100_nllb`)
- `<STRATEGY>` is the realignment strategy, one of:
  - `baseline`: fine-tuning only (no realignment)
  - `before_noaligner`: realignment using sentence-level averaging (proposed method, no word aligner required)
  - `before_fastalign`: realignment using FastAlign word alignments
- `<SEED>` is the random seed (e.g., `17`, `23`, `42`, or `66`)
- `<SELECTION_STRAT>` is the language selection strategy (see below)
- `<TASK>` is the downstream task, one of: `xnli`, `wikiann`, `xtreme_r.udpos`, `xquad`

For example, to reproduce the URIEL featural diversity (10 languages) experiment on XNLI with seed 42:

```bash
bash scripts/2025_aacl/run_langs_selection.sh <DATA_DIR> mix_opus100_nllb before_noaligner 42 most_uriel_en_10 xnli
```

### Language selection strategies

The `<SELECTION_STRAT>` argument controls which languages are used for realignment. Available strategies include:

| Strategy | Description |
|----------|-------------|
| `xt_afri` | All 65 languages (XTREME-R + African languages) |
| `xt_only` | 47 XTREME-R languages only |
| `afri_only` | 21 African languages only |
| `most_uriel_en_{N}` | N most URIEL-featurally diverse languages (computed from English) |
| `least_uriel_en_{N}` | N least URIEL-featurally diverse languages (computed from English) |
| `most_family_en_{N}` | N most phylogenetically diverse languages (excluding Indo-European) |
| `least_family_en_{N}` | N Indo-European-only languages |
| `most_distinct_script_{N}` | N languages from distinct non-Latin scripts |
| `most_same_script_{N}` | N most URIEL-diverse languages restricted to Latin script |
| `least_same_script_{N}` | N least URIEL-diverse Latin-script languages |
| `random_langs_with_seed_{N}` | N randomly selected languages (seeded by `<SEED>`) |
| `abla_random_joshi{CLASS}` | Random pool of languages from Joshi resource class, 10 sampled |
| `abla_most_uriel_joshi{CLASS}` | 10 most URIEL-diverse languages from Joshi class `{CLASS}` |
| `abla_most_family_joshi{CLASS}` | 10 most phylogenetically diverse languages from Joshi class `{CLASS}` |
| `abla_most_script_joshi{CLASS}` | 10 most script-diverse languages from Joshi class `{CLASS}` |
| `abla_random_seen_xlmr` / `abla_random_seen_mbert` | Random pool of languages seen during XLM-R/mBERT pre-training |
| `abla_random_unseen_xlmr` / `abla_random_unseen_mbert` | Random pool of languages unseen during XLM-R/mBERT pre-training |

where `{N}` ∈ {5, 10, 20, 40} and `{CLASS}` ∈ {`2`, `3`, `35`, `45`}.

## Results

In `scripts/2025_aacl/results`, one can find the full results presented in the paper as CSV files, organized by language selection strategy.
