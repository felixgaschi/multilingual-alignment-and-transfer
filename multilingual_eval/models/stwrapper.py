import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Normalize
from sentence_transformers.sentence_transformer.modules import Module


def build_sentence_transformer(
    transformer_model,
    tokenizer,
    max_seq_length=512,
    normalize=False,
):
    """
    Build a SentenceTransformer from an existing transformer object and tokenizer.
    """

    transformer_module = STWrapper(
        transformer_model=transformer_model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    pooling = Pooling(
        word_embedding_dimension=transformer_model.config.hidden_size,
        pooling_mode="mean",
    )

    modules = [transformer_module, pooling]

    if normalize:
        modules.append(Normalize())

    return SentenceTransformer(modules=modules)

class STWrapper(Module):
    def __init__(self, transformer_model, tokenizer, max_seq_length=512):
        super().__init__()
        self.auto_model = transformer_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.embedding_dim = transformer_model.config.hidden_size

    def preprocess(self, inputs, prompt=None, **kwargs):
        # inputs is usually a list of strings from SentenceTransformer.encode()
        if prompt is not None:
            inputs = [prompt + text for text in inputs]

        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

    # for older sentence-transformers compatibility
    def tokenize(self, texts, **kwargs):
        return self.preprocess(texts, **kwargs)

    def forward(self, features, **kwargs):
        outputs = self.auto_model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            return_dict=True,
        )

        features["token_embeddings"] = outputs.last_hidden_state
        return features

    def get_embedding_dimension(self):
        return self.embedding_dim

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        raise NotImplementedError(
            "This wrapper is for in-memory use. Save the HF model/tokenizer separately."
        )