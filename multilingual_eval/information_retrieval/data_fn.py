from pathlib import Path
from typing import Dict, List, Set, Tuple
from datasets import load_dataset


africlir_code_to_name = {
    "afr": "afrikaans",
    "amh": "amharic",
    "ary": "moroccan_arabic",
    "arz": "egyptian_arabic",
    "hau": "hausa",
    "ibo": "igbo",
    "nso": "northern_sotho",
    "sna": "shona",
    "som": "somali",
    "swa": "swahili",
    "tir": "tigrinya",
    "twi": "twi",
    "wol": "wolof",
    "yor": "yoruba",
    "zul": "zulu"
}

clir_matrix_code = [
    'af', 'als', 'am', 'an', 'ar', 'arz', 'ast', 'az', 'azb', 'ba', 'bar', 'be', 'bg', 'bn', 'bpy', 'br', 'bs', 'bug', 
    'ca', 'cdo', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 
    'fa', 'fi', 'fo', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'he', 'hi', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ilo', 
    'io', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'li', 'lmo', 'lt', 'lv', 'mai', 'mg', 
    'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'my', 'mzn', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc', 
    'or', 'os', 'pa', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'ro', 'ru', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 
    'simple', 'sk', 'sl', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tl', 'tr', 'tt', 'uk', 'ur', 
    'uz', 'vec', 'vi', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zh'
    ]

def load_africlirmatrix_eval_data(
    root_dir: str,
    lang: str,
    datasets_cache_dir: str = None,
    version: str = "v1.0",
    hf_dataset_name: str = "castorini/africlirmatrix",
    hf_split: str = "train",
    debug: bool = False,
    debug_num_queries: int = 500,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Dict[str, float]]]:
    """
    Load AfriCLIRMatrix evaluation data.

    Queries/qrels are loaded locally:
        test/queries/topics.africlirmatrix-v1.0.en.{lang}.tsv
        test/qrels/qrels.africlirmatrix-v1.0.en.{lang}.txt

    Documents are loaded from Hugging Face:
        castorini/africlirmatrix, config={africlir_code_to_name[lang]}

    If debug=True:
    - keep only the first debug_num_queries queries
    - keep only qrels for those queries
    - load only documents that appear in those qrels
    """

    root = Path(root_dir)

    queries_path = root / "queries" / f"topics.africlirmatrix-{version}.en.{lang}.tsv"
    qrels_path = root / "qrels" / f"qrels.africlirmatrix-{version}.en.{lang}.txt"

    if not queries_path.exists():
        raise FileNotFoundError(f"Query file not found: {queries_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    # -------------------------
    # 1. Load queries
    # -------------------------
    all_query_ids = []
    query_text_by_id = {}

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            query_id, query_text = line.split("\t", 1)
            all_query_ids.append(query_id)
            query_text_by_id[query_id] = query_text

    if debug:
        selected_query_ids = set(all_query_ids[:debug_num_queries])
    else:
        selected_query_ids = set(all_query_ids)
    print("Num seletected queries: ",len(selected_query_ids))
    
    # -------------------------
    # 2. Load qrels for selected queries
    # -------------------------
    qrels = {}

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
            elif len(parts) == 3:
                query_id, doc_id, relevance = parts
            else:
                raise ValueError(f"Unexpected qrels format: {line}")

            if query_id not in selected_query_ids:
                continue
            relevance = float(relevance)
            if relevance > 0:
                qrels.setdefault(query_id, {})[doc_id] = relevance

    # Keep only queries that have qrels
    query_ids = []
    query_texts = []

    for query_id in all_query_ids:
        if query_id in selected_query_ids and query_id in qrels:
            query_ids.append(query_id)
            query_texts.append(query_text_by_id[query_id])
    print("Number of queries not in qrels: ", len(selected_query_ids)-len(query_ids))

    needed_doc_ids = set()

    for rel_doc_scores in qrels.values():
        needed_doc_ids.update(rel_doc_scores.keys())

    # -------------------------
    # 3. Load documents from HF
    # -------------------------
    docs_dataset = load_dataset(
        hf_dataset_name,
        africlir_code_to_name[lang],
        split=hf_split,
        cache_dir=datasets_cache_dir,
    )

    doc_ids = []
    doc_texts = []

    for row in docs_dataset:
        doc_id = str(row["id"])

        if debug:
            if doc_id not in needed_doc_ids:
                continue

        doc_ids.append(doc_id)
        doc_texts.append(str(row["contents"]))

    if debug:
        loaded_doc_ids = set(doc_ids)

        # Remove qrels whose docs were not found in HF docs
        filtered_qrels = {}

        for query_id, rel_doc_scores in qrels.items():
            kept_rel_doc_scores = {
                doc_id: score
                for doc_id, score in rel_doc_scores.items()
                if doc_id in loaded_doc_ids
            }

            if kept_rel_doc_scores:
                filtered_qrels[query_id] = kept_rel_doc_scores

        qrels = filtered_qrels

        # Remove queries whose relevant docs were not found
        filtered_query_ids = []
        filtered_query_texts = []

        for query_id, query_text in zip(query_ids, query_texts):
            if query_id in qrels:
                filtered_query_ids.append(query_id)
                filtered_query_texts.append(query_text)

        query_ids = filtered_query_ids
        query_texts = filtered_query_texts

        print(
            f"[DEBUG AfriCLIRMatrix] Loaded {len(query_ids)} queries, "
            f"{len(doc_ids)} docs, "
            f"{len(qrels)} qrel query entries."
        )

    return doc_ids, doc_texts, query_ids, query_texts, qrels

def load_clirmatrix_eval_data(
    doc_lang,
    query_lang="en",
    split="test1",
    subset="bi139-base",
    debug=False,
    debug_num_queries=500,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Dict[str, float]]]:
    """
    Example dataset id:
        clirmatrix/en/bi139-base/de/test1

    This means:
    - documents are English
    - queries are German
    - cross-lingual retrieval: German query -> English document

    If debug=True:
    - load only the first debug_num_docs documents
    - keep only qrels whose relevant docs are inside those documents
    - keep only queries that still have at least one relevant doc
    """
    import ir_datasets

    dataset_id = f"clirmatrix/{doc_lang}/{subset}/{query_lang}/{split}"
    dataset = ir_datasets.load(dataset_id)

    # -------------------------
    # 1. Load queries
    # -------------------------
    all_query_ids = []
    query_text_by_id = {}

    for query in dataset.queries_iter():
        all_query_ids.append(query.query_id)
        query_text_by_id[query.query_id] = query.text

    if debug:
        selected_query_ids = set(all_query_ids[:debug_num_queries])
    else:
        selected_query_ids = set(all_query_ids)

    # -------------------------
    # 2. Load qrels for selected queries
    # -------------------------
    qrels = {}

    for qrel in dataset.qrels_iter():
        if qrel.query_id not in selected_query_ids:
            continue
        
        rel = float(qrel.relevance)
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = rel

    query_ids = []
    query_texts = []

    for query_id in all_query_ids:
        if query_id in selected_query_ids and query_id in qrels:
            query_ids.append(query_id)
            query_texts.append(query_text_by_id[query_id])

    needed_doc_ids = set()

    for rel_doc_scores in qrels.values():
        needed_doc_ids.update(rel_doc_scores.keys())

    # -------------------------
    # 3. Load docs
    # -------------------------
    doc_ids = []
    doc_texts = []

    if debug:
        for doc in dataset.docs_iter():
            if doc.doc_id in needed_doc_ids:
                doc_ids.append(doc.doc_id)
                doc_texts.append(doc.text)

            if len(doc_ids) >= len(needed_doc_ids):
                break
    else:
        for doc in dataset.docs_iter():
            doc_ids.append(doc.doc_id)
            doc_texts.append(doc.text)

    if debug:
        loaded_doc_ids = set(doc_ids)

        filtered_qrels = {}

        for query_id, rel_doc_scores in qrels.items():
            kept_rel_doc_scores = {
                doc_id: score
                for doc_id, score in rel_doc_scores.items()
                if doc_id in loaded_doc_ids
            }

            if kept_rel_doc_scores:
                filtered_qrels[query_id] = kept_rel_doc_scores

        qrels = filtered_qrels

        filtered_query_ids = []
        filtered_query_texts = []

        for query_id, query_text in zip(query_ids, query_texts):
            if query_id in qrels:
                filtered_query_ids.append(query_id)
                filtered_query_texts.append(query_text)

        query_ids = filtered_query_ids
        query_texts = filtered_query_texts

        print(
            f"[DEBUG CLIRMatrix] Loaded {len(query_ids)} queries, "
            f"{len(doc_ids)} docs, "
            f"{len(qrels)} qrel query entries."
        )

    return doc_ids, doc_texts, query_ids, query_texts, qrels

def load_clir_eval_data(
    doc_lang: str,
    *,
    africlir_root_dir: str,
    datasets_cache_dir: str = None,
    query_lang: str = "en",
    split: str = "test1",
    clirmatrix_subset: str = "bi139-base",
    africlir_version: str = "v1.0",
    africlir_hf_dataset_name: str = "castorini/africlirmatrix",
    africlir_hf_split: str = "train",
    debug: bool = False,
    debug_num_queries: int = 500,
):
    doc_lang = doc_lang.strip()

    if doc_lang in africlir_code_to_name:
        doc_ids, doc_texts, query_ids, query_texts, qrels = load_africlirmatrix_eval_data(
            root_dir=africlir_root_dir,
            lang=doc_lang,
            datasets_cache_dir=datasets_cache_dir,
            version=africlir_version,
            hf_dataset_name=africlir_hf_dataset_name,
            hf_split=africlir_hf_split,
            debug=debug,
            debug_num_queries=debug_num_queries,
        )

        dataset_name = "AfriCLIRMatrix"

    elif doc_lang in clir_matrix_code:
        doc_ids, doc_texts, query_ids, query_texts, qrels = load_clirmatrix_eval_data(
            doc_lang=doc_lang,
            query_lang=query_lang,
            split=split,
            subset=clirmatrix_subset,
            debug=debug,
            debug_num_queries=debug_num_queries,
        )

        dataset_name = "CLIRMatrix"

    else:
        raise ValueError(
            f"Unsupported document language code: {doc_lang}. "
            f"Not found in africlir_code_to_name or clir_matrix_code."
        )

    return {
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "query_ids": query_ids,
        "query_texts": query_texts,
        "qrels": qrels,
        "dataset_name": dataset_name,
        "query_lang": query_lang,
        "doc_lang": doc_lang,
    }