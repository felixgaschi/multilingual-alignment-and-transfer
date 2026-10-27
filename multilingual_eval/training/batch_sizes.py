model_to_batch_size_small = {
    "xlm-roberta-base": 32,
    "bert-base-multilingual-cased": 32,
    "distilbert-base-multilingual-cased": 32,
    "xlm-roberta-large": 32,
    "google/gemma-2-2b": 1,
    "google/gemma-2-9b": 1,
    "meta-llama/Llama-3.2-3B": 1,
    "meta-llama/Llama-3.1-8B": 1,
}

model_to_batch_size_big = {
    "xlm-roberta-base": 128,
    "bert-base-multilingual-cased": 128,
    "distilbert-base-multilingual-cased": 128,
    "xlm-roberta-large": 128,
    "google/gemma-2-2b": 32,
    "google/gemma-2-9b": 4,
    "meta-llama/Llama-3.2-3B": 16,
    "meta-llama/Llama-3.1-8B": 4,
}


def get_batch_size(model_name, real_batch_size=32, large_gpu=False):
    """
    Empirical heuristics to get batch size according to model name
    in order to avoid OOM. This is very device-specific, and should not
    be used as is
    """
    if large_gpu:
        return min(model_to_batch_size_big.get(model_name, real_batch_size), real_batch_size)
    return min(model_to_batch_size_small.get(model_name, 4), real_batch_size)
