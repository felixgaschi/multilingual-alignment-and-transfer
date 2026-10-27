import torch
from datasets import load_dataset
from sentence_transformers import losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer


def supports_bf16():
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()

    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def train_clir(
    model,
    datasets_cache_dir=None,
    dataset_name="bclavie/msmarco-10m-triplets",
    dataset_split="train",
    query_column="query",
    positive_column="positive",
    **training_kwargs,
):
    """
    Fine-tune a SentenceTransformer model on MS MARCO triplets.
    """

    train_dataset = load_dataset(dataset_name, cache_dir=datasets_cache_dir)[dataset_split]

    if query_column != "anchor":
        train_dataset = train_dataset.rename_column(query_column, "anchor")

    # MultipleNegativesRankingLoss expects paired columns such as:
    # anchor + positive
    train_loss = losses.MultipleNegativesRankingLoss(model)

    use_bf16 = supports_bf16()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    default_args = {
        "max_steps": 1000,
        "per_device_train_batch_size": 64,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_steps": 200,
        "save_strategy": "no",
        "report_to": "none",
    }

    # User-provided kwargs override defaults
    default_args.update(training_kwargs)

    args = SentenceTransformerTrainingArguments(**default_args)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()

    return model