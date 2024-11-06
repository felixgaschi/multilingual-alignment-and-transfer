# AlignFreeze continuation

WORK-IN-PROGRESS

## Structure

The code for the project must be put into this folder: `scripts/2025_alignfreeze_continuation`

For now, we are exploring different leads, so we'll keep them separate in different folders:

- `unknown_token` for looking at the impact of the amount of [UNK] token
- `regression` for continuing the experiments about regression
- `alignfreeze` for continuing experiments like AlignFreeze (like single-layer realignment)
- `distillation` for work investigating the impact of distillation
- `representation_analysis` for trying to deconstruct multilingual representations built by multilingual models

## The data

Relevant data is stored in `data`, with several subdirectories:

- `realignment_results`: with CSV files containing the result of previous realignment and cross-lingual transfer experiments.
- `linguistic_features`: with CSV containing intersting linguistic features

### realignment result files

The CSV files contain the following properties:

- seed: random seed used for model initialization and data shuffling
- model
- task
- method: realignment method used
  - baseline: simple fine-tuning (no realignemnt)
  - `<moment>_<aligner` for full realignment (without freezing) with aligner being the method of word pair extraction (fastalign, awesome or dico) and moment being either:
    - before: the normal way to perform realignment (realignment first, fine-tuning after)
    - during: another way to do it where realignment is performed simultaneously with fine-tuning (loss are summed per batched and backpropagation is performed on the sum). For now, we focus on the before method
  - `freeze_realign_unfreeze_<aligner>` for AlignFreeze with front-freezing
  - `freeze_realign_unfreeze_last_half_<aligner>` for Alignfreeze `ith back-freezin`
  - there are other methods (where we freeze during reali`nment in the "during" approach) but they should be discarded for now
- `finetuning_steps` ,`realignment_steps` , `distinct_realignment_samples`, `repeated_realignment_samples`, `train_loss`, `realignment_loss`, `task_loss`: information about realignment and fine-tuning
- `eval_<lang>_accuracy` and `final_<lang>_accuracy`: test accuracy in the given language, both values should actually be the same
- `final_eval_avg_accuracy`: average accuracy over target languages
- `final_eval_same_accuracy`: test accuracy in the training language (i.e. English)

Not all methods were applied to all languages. The main methods to look at (which has been used for all models and tasks) are:

- baseline
- before_dico
- freeze_realign_unfreeze_dico
- freeze_realign_unfreeze_last_half_dico

### Linguistic features CSV

Source: https://docs.google.com/spreadsheets/d/1sJvrYL5kHRrZG7JPzeDiK9z9k7pReKRZGEll-NmbmMQ/edit?gid=1927739942#gid=1927739942

Each csv in `data/linguistic_features` concerns one specific feature. The two columns of interest for us in those files are:

- `wals code` the language code, which is a 3-letter code that can be converted in the 2-letter code (if it exists) used in the rest of the data with the function `get_alpha2_from_alpha3` in `scripts/2025_alignfreeze_continuation/utils/language_names.py`
- `value` which contains the value for the categorical feature
