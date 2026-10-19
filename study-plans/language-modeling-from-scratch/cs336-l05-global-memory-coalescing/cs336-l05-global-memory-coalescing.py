import numpy as np

def coalescing_analysis(byte_addresses, access_width_bytes=4, cache_line_bytes=128):
    """
    Returns: dictionary containing line IDs, transaction count, and useful-byte fraction
    """
    addresses = np.asarray(byte_addresses, dtype=np.int64)
    touched_lines = []
    requested_bytes = set()
    for address in addresses:
        start = int(address)
        first_line = start // cache_line_bytes
        last_line = (start + access_width_bytes - 1) // cache_line_bytes
        touched_lines.extend(range(first_line, last_line + 1))
        requested_bytes.update(range(start, start + access_width_bytes))

    line_ids = np.array(sorted(set(touched_lines)), dtype=np.int64)
    transaction_count = int(line_ids.size)
    useful_byte_fraction = float(
        len(requested_bytes) / (transaction_count * cache_line_bytes)
    )
    return {
        "line_ids": line_ids,
        "transaction_count": transaction_count,
        "useful_byte_fraction": useful_byte_fraction,
    }
