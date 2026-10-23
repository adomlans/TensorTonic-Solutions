import hashlib
import unicodedata

def stable_exact_deduplication(documents, lowercase=True,
                               collapse_whitespace=True, hash_bits=64):
    """
    Returns: dictionary containing retained IDs and duplicate ownership
    """
    retained_ids = []
    removed_to_retained = {}
    buckets = {}
    mask = (1 << hash_bits) - 1
    for document in documents:
        normalized = unicodedata.normalize("NFKC", document["text"])
        if lowercase:
            normalized = normalized.casefold()
        if collapse_whitespace:
            normalized = " ".join(normalized.split())
        digest = int.from_bytes(
            hashlib.sha256(normalized.encode("utf-8")).digest(), "big"
        ) & mask
        owner = None
        for retained_text, retained_id in buckets.get(digest, []):
            if retained_text == normalized:
                owner = retained_id
                break
        if owner is None:
            retained_ids.append(document["id"])
            buckets.setdefault(digest, []).append((normalized, document["id"]))
        else:
            removed_to_retained[document["id"]] = owner
    return {
        "retained_ids": retained_ids,
        "removed_to_retained": removed_to_retained,
    }