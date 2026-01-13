# Multilingual alignment and transfer

This repository holds the code for the following papers:

- [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076v1), Félix Gaschi, François Plesse, Parisa Rastin, Yannick Toussaint. (IJCNN 2022)
- [Exploring the Relationship between Alignment and Cross-lingual Transfer in Multilingual Transformers](https://aclanthology.org/2023.findings-acl.189/), Félix Gaschi, Patricio Cerda, Parisa Rastin, Yannick Toussaint. (Findings of ACL 2023)
- [ALIGNFREEZE: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages](https://aclanthology.org/2025.naacl-short.48/), Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee. (NAACL 2025)
- [Rethinking what Matters: Effective and Robust Multilingual Realignment for Low-Resource Languages](https://aclanthology.org/2025.ijcnlp-long.102/), Quang Phuoc Nguyen\*, David Anugraha\*, Felix Gaschi\*, Jun Bin Cheng, En-Shiun Annie Lee. (IJCNLP-AACL 2025)

Note: this repository is the direct successor of the [Multilingual Alignment and Transfer repository](https://github.com/posos-tech/ml-pipeline/). It is extended to contain the code for more recent works.

## Architecture of the repository

- `download_resources` contain scripts to download necessary resources
- `multilingual_eval` contain the source code
- `scripts` contains launchables for reproducing experiments
- `subscripts` contain various scripts for using external dependencies (e.g. Stansford segmenter) and preparing data (sampling dataset, import results from wandb etc...)

## How to use the repository

The reusable source code is found in `multilingual_eval`, while paper-specific scripts that allows to reproduce a specific experiments and figures from a given paper are found in dedicated subdirectories of `scripts`:

- [scripts/2022_ijcnn](scripts/2022_ijcnn/README.md) for IJCNN 2022
- [scripts/2023_acl](scripts/2023_acl/README.md) for Findings of ACL 2023
- [scripts/2025_naacl](scripts/2025_naacl/README.md) for NAACL 2025
- [scripts/2025_aacl](scripts/2025_aacl/README.md) for IJCNLP-AACL 2025
