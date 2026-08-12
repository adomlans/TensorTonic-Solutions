import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    results = []
    for spec in pair_specs:
        sent_a = documents[spec["doc_a"]][spec["sent_a"]]
        sent_b = documents[spec["doc_b"]][spec["sent_b"]]
        if spec["doc_a"] == spec["doc_b"] and spec["sent_b"] == spec["sent_a"] + 1:
            label = 1
        else:
            label = 0
        results.append((sent_a, sent_b, label))
    return results

class NSPHead:
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        return cls_hidden @ self.W + self.b

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
