import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.zeros(0, dtype=float)
    doc_lens = np.array([len(d) for d in docs], dtype=float)
    avgdl = float(doc_lens.mean()) if N > 0 else 0.0
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    doc_tfs = [Counter(d) for d in docs]
    q_terms = list(dict.fromkeys(query_tokens))
    scores = np.zeros(N, dtype=float)
    for t in q_terms:
        dfi = df.get(t, 0)
        if dfi == 0:
            idf = 0.0
        else:
            idf = math.log((N - dfi + 0.5) / (dfi + 0.5) + 1.0)
        if idf == 0.0:
            continue
        tf_vec = np.array([doc_tfs[i].get(t, 0) for i in range(N)], dtype=float)
        denom = tf_vec + k1 * (1.0 - b + b * (doc_lens / (avgdl if avgdl > 0 else 1.0)))
        with np.errstate(divide='ignore', invalid='ignore'):
            term_score = idf * ((tf_vec * (k1 + 1.0)) / np.where(denom == 0, 1.0, denom))
            term_score = np.nan_to_num(term_score, nan=0.0, posinf=0.0, neginf=0.0)
        scores += term_score
    return scores
