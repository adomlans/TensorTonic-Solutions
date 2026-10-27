import copy

def backup_mcts_path(path, leaf_value, leaf_player):
    """
    Returns: Updated root-to-leaf path records with incremented N, W, and Q values.
    """
    updated = copy.deepcopy(path)
    current_player = leaf_player
    current_value = float(leaf_value)

    for record in reversed(updated):
        if record["player"] != current_player:
            current_value = -current_value

        record["N"] += 1
        record["W"] = float(record["W"]) + current_value
        record["Q"] = record["W"] / record["N"]
        record["P"] = float(record["P"])
        current_player = record["player"]

    return updated