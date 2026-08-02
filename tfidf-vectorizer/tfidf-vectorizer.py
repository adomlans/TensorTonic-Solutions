import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    tokens_list = [doc.lower().split() for doc in documents]
    N = len(documents)
    vocab = sorted(set(t for tokens in tokens_list for t in tokens))
    V = len(vocab)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    if N == 0 or V == 0:
        return np.zeros((N, 0), dtype=float), []
    df = Counter()
    for tokens in tokens_list:
        for t in set(tokens):
            df[t] += 1
    matrix = np.zeros((N, V), dtype=float)
    for i, tokens in enumerate(tokens_list):
        tf_counts = Counter(tokens)
        doc_len = len(tokens)
        if doc_len == 0:
            continue
        for t, count in tf_counts.items():
            j = vocab_idx[t]
            tf = count / doc_len
            idf = math.log(N / df[t])
            matrix[i, j] = tf * idf
    return matrix, vocab
