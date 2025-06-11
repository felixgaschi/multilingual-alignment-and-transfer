from tqdm import tqdm
import logging
from itertools import combinations
from math import comb
from multiprocessing import Pool, cpu_count

import numpy as np
from urielplus import urielplus

logging.getLogger().setLevel(logging.WARNING)

LANGUAGES_DICT: dict[str, str] = {
    'stan1318': "Standard Arabic",
    'hebr1245': "Hebrew",
    'nucl1301': "Turkish",
    'west2369': "Western Farsi (Persian)",
    'hind1269': "Hindi",
    'latv1249': "Latvian",
    'lith1251': "Lithuanian",
    'czec1258': "Czech",
    'poli1260': "Polish",
    # 'slov1269': "Slovak",
    'bulg1262': "Bulgarian",
    'slov1268': "Slovenian",
    'russ1263': "Russian",
    'ukra1253': "Ukrainian",
    'mode1248': "Greek",
    'stan1290': "French",
    'stan1288': "Spanish",
    'stan1289': "Catalan",
    'ital1282': "Italian",
    'port1283': "Portuguese",
    'roma1327': "Romanian",
    'dani1285': "Danish",
    'norw1258': "Norwegian",
    'swed1254': "Swedish",
    'stan1295': "German",
    'afri1274': "Afrikaans",
    'finn1318': "Finnish",
    'hung1274': "Hungarian",
    'mand1415': "Mandarin (Chinese)",
    'thai1261': "Thai",
    'viet1252': "Vietnamese",
    'nucl1643': "Japanese",
    'kore1280': "Korean",
    'tami1289': "Tamil",
}

def _worker_chunk(args):
    """
    Worker function that scans only those combinations whose first index i is in [i_start, i_end).
    args = (i_start, i_end, L, n, distances)
    - distances: dict[(i,j)] -> float, precomputed for all i<j
    """
    i_start, i_end, num_langs, n, distances = args

    best_score = -np.inf
    best_indices: tuple[int, ...] = ()

    # For every possible "first index" = i in our subrange:
    for i in range(i_start, min(i_end, num_langs - n + 1)):
        # build combinations of the remaining n-1 positions from [i+1 .. num_langs-1]
        for tail in combinations(range(i + 1, num_langs), n - 1):
            # tail is a tuple of length (n-1); full combo is (i,) + tail
            s = 0.0
            # sum pairwise distances within (i, *tail)
            # there are binom(n,2) pairs to add
            for idx_j, idx_k in combinations((i,) + tail, 2):
                # ensure idx_j < idx_k for lookup
                a, b = (idx_j, idx_k) if idx_j < idx_k else (idx_k, idx_j)
                s += distances[(a, b)]

            if s > best_score:
                best_score = s
                best_indices = (i,) + tail

    return best_score, best_indices


def max_diverse_subset_parallel(
    languages: list[str],
    n: int,
    distance_fn,
    num_workers: int = None
) -> tuple[list[int], list[str], float]:
    """
    Parallel version of max_diverse_subset_precomputed. Splits the "first index i" range
    into num_workers chunks, runs each chunk in a separate process, then picks the overall best.

    Args:
      - languages: list of language-IDs (len num_langs)
      - n: size of subset to pick
      - distance_fn: function signature distance_fn(str, lang1, lang2) -> float
      - num_workers: how many processes; default = cpu_count()

    Returns: (best_indices_list, best_subset_keys, best_score)
    """

    num_langs = len(languages)
    if n > num_langs:
        raise ValueError(f"n={n} > number of languages={num_langs}")

    # 2) Decide how many worker processes
    if num_workers is None:
        num_workers = cpu_count()
    num_workers = min(num_workers, num_langs - n + 1)  # no need for more workers than possible i-values

    # 3) Partition the range of valid "first indices" [0 .. num_langs-n] into chunks
    max_first = num_langs - n + 1  # i can go from 0 up to num_langs-n
    chunk_size = (max_first + num_workers - 1) // num_workers  # ceil division

    pool_args = []
    for w in range(num_workers):
        i_start = w * chunk_size
        i_end = min((w + 1) * chunk_size, max_first)
        if i_start >= i_end:
            break
        pool_args.append((i_start, i_end, num_langs, n, distances))

    # 4) Launch Pool; each worker returns (best_score, best_indices) for its chunk
    with Pool(processes=len(pool_args)) as pool:
        results = list(tqdm(pool.imap_unordered(_worker_chunk, pool_args), total=len(pool_args), desc="Workers done"))

    # 5) Find global best among all workers
    global_best_score = -np.inf
    global_best_indices: tuple[int, ...] = ()
    for score, indices_tuple in results:
        if score > global_best_score:
            global_best_score = score
            global_best_indices = indices_tuple

    # 6) Convert indices → language keys
    best_subset_keys = [languages[i] for i in global_best_indices]
    return list(global_best_indices), best_subset_keys, global_best_score

if __name__ == '__main__':
    uriel = urielplus.URIELPlus()
    uriel.integrate_databases()
    uriel.set_aggregation('U')
    uriel.aggregate()
    uriel.softimpute_imputation()

    # 1) Build a dict of distances so we never recompute distance_fn for the same pair.
    #    We'll store distances[(i,j)] for i < j.
    lang_list = list(LANGUAGES_DICT.keys())
    num_langs = len(lang_list)
    distances: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(num_langs), 2):
        # e.g. "featural" is hard-coded here—swap in whatever your real signature is.
        distances[(i, j)] = uriel.new_distance('featural', lang_list[i], lang_list[j])

    n = 3
    res = max_diverse_subset_parallel(lang_list, n, uriel.new_distance)
    print(f"Most Diverse Subset for {n} languages:")
    print([LANGUAGES_DICT[code] for code in res[1]])
    print("Dissimilarity score:")
    print(res[2])
    print("Indices:")
    print(res[0])
    
    n = 7
    res = max_diverse_subset_parallel(lang_list, n, uriel.new_distance)
    print(f"Most Diverse Subset for {n} languages:")
    print([LANGUAGES_DICT[code] for code in res[1]])
    print("Dissimilarity score:")
    print(res[2])
    print("Indices:")
    print(res[0])

    n = 14
    res = max_diverse_subset_parallel(lang_list, n, uriel.new_distance)
    print(f"Most Diverse Subset for {n} languages:")
    print([LANGUAGES_DICT[code] for code in res[1]])
    print("Dissimilarity score:")
    print(res[2])
    print("Indices:")
    print(res[0])
    
    n = 28
    res = max_diverse_subset_parallel(lang_list, n, uriel.new_distance)
    print(f"Most Diverse Subset for {n} languages:")
    print([LANGUAGES_DICT[code] for code in res[1]])
    print("Dissimilarity score:")
    print(res[2])
    print("Indices:")
    print(res[0])
