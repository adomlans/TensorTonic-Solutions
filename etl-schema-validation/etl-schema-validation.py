def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    TYPE_CHECK = {
        "int": lambda v: type(v) == int,
        "float": lambda v: type(v) in (int, float),
        "str": lambda v: type(v) == str,
    }
    results = []
    for idx, record in enumerate(records):
        errors = []
        for col_def in schema:
            col = col_def["column"]
            if col not in record:
                errors.append(f"{col}: missing")
                continue
            value = record[col]
            if value is None:
                if not col_def["nullable"]:
                    errors.append(f"{col}: null")
                continue
            if not TYPE_CHECK[col_def["type"]](value):
                errors.append(f"{col}: expected {col_def['type']}, got {type(value).__name__}")
                continue
            if "min" in col_def and value < col_def["min"]:
                errors.append(f"{col}: out of range")
            elif "max" in col_def and value > col_def["max"]:
                errors.append(f"{col}: out of range")
        results.append((idx, len(errors) == 0, errors))
    return results
