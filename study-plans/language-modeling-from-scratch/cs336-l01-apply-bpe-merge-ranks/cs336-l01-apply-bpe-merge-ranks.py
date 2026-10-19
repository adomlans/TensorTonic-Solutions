def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns: list of token ids after applying every merge in merges, in order, to the UTF-8 byte encoding of text
    """
    ids = list(text.encode('utf-8'))
    for (left_id, right_id, new_id) in merges:
        if len(ids) < 2:
            continue
        merged = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == left_id and (ids[i + 1] == right_id):
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged
    return ids

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns: the string obtained by concatenating each id's bytes from vocab, in order, and UTF-8-decoding the result
    """
    raw_bytes = []
    for token_id in ids:
        raw_bytes.extend(vocab[token_id])
    return bytes(raw_bytes).decode('utf-8')
