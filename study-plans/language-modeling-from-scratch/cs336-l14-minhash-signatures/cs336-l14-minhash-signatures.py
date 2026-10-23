import hashlib
import json

import numpy as np

def minhash_signatures(documents, shingle_size, num_hashes, seed):
    """
    Returns: dictionary containing signature and estimated-similarity arrays
    """
    shingle_sets = []
    for document in documents:
        shingles = {
            json.dumps(
                document[index:index + shingle_size],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for index in range(max(0, len(document) - shingle_size + 1))
        }
        shingle_sets.append(shingles)

    sentinel = (1 << 32) - 1
    signatures = np.full(
        (len(documents), num_hashes), sentinel, dtype=np.uint32
    )
    for row, shingles in enumerate(shingle_sets):
        for column in range(num_hashes):
            prefix = f"{seed}:{column}:".encode("utf-8")
            for shingle in shingles:
                digest = hashlib.sha256(
                    prefix + shingle.encode("utf-8")
                ).digest()
                value = int.from_bytes(digest[:4], "big")
                if value < signatures[row, column]:
                    signatures[row, column] = value

    similarities = np.eye(len(documents), dtype=np.float64)
    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
            if not shingle_sets[left] and not shingle_sets[right]:
                estimate = 1.0
            elif not shingle_sets[left] or not shingle_sets[right]:
                estimate = 0.0
            else:
                estimate = float(np.mean(signatures[left] == signatures[right]))
            similarities[left, right] = estimate
            similarities[right, left] = estimate
    return {"signatures": signatures, "similarities": similarities}
