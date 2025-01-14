# Single-layer realignment

To perform single layer realignment, you need to launch one of the batch script inside this directory from the root of this repository.

For example, for performing single-layer realignment with udpos, you need to launch:

```
bash scripts/2025_alignfreeze_continuation/alignfreeze/single_layer_realignment_udpos.sh <DATA_DIR> <MODEL_PATH> "<OPTIONS>"
```

With parameters:

- DATA_DIR: the path where everything will be stored (from the realignment dataset to the results)
- MODEL_PATH: the path of the model (either a HF model or a local path)
- OPTIONS: any option to pass to the python script (e.g. "--large_gpu --seed 61"). Notable options are:
    - `--debug` if added, it will run a short version of the script (with a reduced dataset) to "quickly" test it
    - `--large_gpu` if you have a GPU bigger than 8BG you might want to try this one to go faster, but remove it if you get an OOM
    - `--n_seeds` (default to 5) number of seeds to test, you can use up to 10
    - `--seeds` if you want to use specific seeds instead of the default ones you can specify them (e.g. `--seeds 42 30 4`)

More options can be found directly in `scripts/2023_acl/controlled_realignment.py` or by running `python scripts/2023_acl/controlled_realignment.py --help` but some of them are already set in `scripts/2025_alignfreeze_continuation/alignfreeze/single_layer_realignment_udpos.sh`

## Requirements

This has been tested with Python 3.9.

There are two sets of Python requirements to install:

- scripts/2023_acl/requirements.txt
- scripts/2024_emnlp/additional_requirements.txt (might or might not be required)

There is a bunch of things to download, the following script should allow to get everything needed:

```
bash scripts/2025_alignfreeze_continuation/alignfreeze/download_resources.sh DATA_DIR
```

with DATA_DIR being the same path that you will use in the running script. It should be in a disk with some available space (It can take up to 80Go).