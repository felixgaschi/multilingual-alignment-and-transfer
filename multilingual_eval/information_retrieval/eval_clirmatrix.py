import os
import json
import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from ranx import Qrels, Run, evaluate

from multilingual_eval.information_retrieval.data_fn import load_clir_eval_data


def make_cache_dir(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def safe_cache_name(*parts):
    return "_".join(str(p).replace("/", "-") for p in parts)

def encode_or_load_doc_embeddings(
    model,
    doc_ids,
    doc_texts,
    cache_dir,
    cache_prefix,
    batch_size=128,
    device="cuda",
    use_cache=True,
):
    cache_dir = make_cache_dir(cache_dir)

    ids_path = cache_dir / f"{cache_prefix}_doc_ids.npy"
    emb_path = cache_dir / f"{cache_prefix}_doc_embeddings.npy"

    if use_cache and ids_path.exists() and emb_path.exists():
        print(f"Loading cached document embeddings: {emb_path}")

        cached_doc_ids = np.load(ids_path, allow_pickle=True).tolist()
        doc_embeddings = np.load(emb_path).astype("float32")

        return cached_doc_ids, doc_embeddings

    print(f"Encoding {len(doc_texts)} documents...")

    doc_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    ).astype("float32")

    if use_cache:
        print(f"Saving document embedding cache to: {emb_path}")
        np.save(ids_path, np.array(doc_ids, dtype=object))
        np.save(emb_path, doc_embeddings)

    return doc_ids, doc_embeddings

def build_faiss_index(doc_embeddings):
    import faiss

    print(f"Building FAISS index for {doc_embeddings.shape[0]} documents...")

    dim = doc_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype("float32"))

    return index

def evaluate_ndcg10_faiss(
    model,
    doc_ids,
    doc_texts,
    query_ids,
    query_texts,
    qrels,
    cache_dir,
    cache_prefix,
    batch_size=128,
    device="cuda",
    use_cache=True,
):
    cached_doc_ids, doc_embeddings = encode_or_load_doc_embeddings(
        model=model,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        batch_size=batch_size,
        device=device,
        use_cache=use_cache,
    )

    index = build_faiss_index(doc_embeddings=doc_embeddings)

    print(f"Encoding {len(query_texts)} queries...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    ).astype("float32")

    # FAISS returns both scores and document indices
    top_scores, top_indices = index.search(query_embeddings, 10)

    # -------------------------
    # 1. Build ranx qrels
    # -------------------------
    ranx_qrels_dict = {
        query_id: {
            doc_id: relevance
            for doc_id, relevance in doc_rels.items()
        }
        for query_id, doc_rels in qrels.items()
        if doc_rels
    }

    ranx_qrels = Qrels(ranx_qrels_dict)

    # -------------------------
    # 2. Build ranx run from FAISS top-10
    # -------------------------
    run_dict = {}

    for query_idx, query_id in enumerate(query_ids):
        query_id = str(query_id)

        # Optional: evaluate only queries that have qrels
        if query_id not in ranx_qrels_dict:
            continue

        retrieved_docs = {}

        for rank_idx, doc_idx in enumerate(top_indices[query_idx]):
            doc_id = str(cached_doc_ids[doc_idx])
            score = float(top_scores[query_idx][rank_idx])

            retrieved_docs[doc_id] = score

        run_dict[query_id] = retrieved_docs

    ranx_run = Run(run_dict)

    # -------------------------
    # 3. Evaluate NDCG@10
    # -------------------------
    ndcg10 = evaluate(
        qrels=ranx_qrels,
        run=ranx_run,
        metrics=["ndcg@10"],
    )

    return {
        "nDCG@10": float(ndcg10),
        "num_queries": len(run_dict),
    }

def run_clir_eval(
    model,
    doc_lang,
    africlir_root_dir,
    datasets_cache_dir=None,
    query_lang="en",
    split="test1",
    clirmatrix_subset="bi139-base",
    batch_size=128,
    device=None,
    cache_dir="clir_cache",
    use_cache=False,
    debug: bool = False,
    debug_num_queries: int = 500,
):
    """
    Load CLIRMatrix or AfriCLIRMatrix data and evaluate nDCG@10.

    If use_cache=True:
        loads/saves document embeddings.

    If use_cache=False:
        encodes documents fresh.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_clir_eval_data(
        doc_lang=doc_lang,
        africlir_root_dir=africlir_root_dir,
        datasets_cache_dir=datasets_cache_dir,
        query_lang=query_lang,
        split=split,
        clirmatrix_subset=clirmatrix_subset,
        debug=debug,
        debug_num_queries=debug_num_queries
    )

    dataset_name = data["dataset_name"]

    if dataset_name == "CLIRMatrix":
        cache_prefix = safe_cache_name(
            "CLIRMatrix",
            data["doc_lang"],
            clirmatrix_subset,
        )

    elif dataset_name == "AfriCLIRMatrix":
        cache_prefix = safe_cache_name(
            "AfriCLIRMatrix",
            data["doc_lang"],
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print("=" * 80)
    print(f"Dataset: {data['dataset_name']}")
    print(f"Query language: {data['query_lang']}")
    print(f"Document language: {data['doc_lang']}")
    print(f"Documents: {len(data['doc_texts'])}")
    print(f"Queries: {len(data['query_texts'])}")
    print(f"Qrels: {len(data['qrels'])}")
    print(f"Cache enabled: {use_cache}")
    print(f"Cache prefix: {cache_prefix}")

    scores = evaluate_ndcg10_faiss(
        model=model,
        doc_ids=data["doc_ids"],
        doc_texts=data["doc_texts"],
        query_ids=data["query_ids"],
        query_texts=data["query_texts"],
        qrels=data["qrels"],
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        batch_size=batch_size,
        device=device,
        use_cache=use_cache,
    )

    return {
        "dataset_name": data["dataset_name"],
        "query_lang": data["query_lang"],
        "doc_lang": data["doc_lang"],
        "cache_prefix": cache_prefix if use_cache else None,
        "use_cache": use_cache,
        **scores,
    }

def run_clir_eval_many(
    model,
    doc_langs,
    africlir_root_dir,
    datasets_cache_dir=None,
    query_lang="en",
    split="test1",
    clirmatrix_subset="bi139-base",
    batch_size=128,
    cache_dir="clir_cache",
    use_cache=False,
    debug: bool = False,
    debug_num_queries: int = 500,
):
    all_results = {}

    for lang in doc_langs:
        print("=" * 80)
        print(f"Evaluating language: {lang}")

        try:
            result = run_clir_eval(
                model=model,
                doc_lang=lang,
                africlir_root_dir=africlir_root_dir,
                datasets_cache_dir=datasets_cache_dir,
                query_lang=query_lang,
                split=split,
                clirmatrix_subset=clirmatrix_subset,
                batch_size=batch_size,
                cache_dir=cache_dir,
                use_cache=use_cache,
                debug=debug,
                debug_num_queries=debug_num_queries,
            )

            result["status"] = "ok"
            all_results[f"eval_{lang}"] = result["nDCG@10"]

        except Exception as e:
            result = {
                "doc_lang": lang,
                "dataset_name": None,
                "nDCG@10": None,
                "num_queries": 0,
                "status": "failed",
                "error": str(e),
                "use_cache": use_cache,
            }

        print(result)

    all_results['eval_avg'] = np.mean(list(all_results.values()))

    return all_results