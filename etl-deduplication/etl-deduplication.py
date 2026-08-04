def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    groups = {}
    order = []
    for record in records:
        key = tuple(record[col] for col in key_columns)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    result = []
    for key in order:
        group = groups[key]
        if strategy == "first":
            result.append(group[0])
        elif strategy == "last":
            result.append(group[-1])
        elif strategy == "most_complete":
            result.append(min(group, key=lambda r: sum(1 for v in r.values() if v is None)))
    return result
