import math
import numpy as np

def neural_guided_mcts(root_state, transitions, legal_actions, players, policy_logits, value_predictions, terminal_values, simulations, cpuct, temperature):
    actions_by_state = {state: list(actions) for state, actions in legal_actions.items()}
    priors = {}
    for state, actions in actions_by_state.items():
        logits = np.asarray(policy_logits[state], dtype=np.float64)
        weights = np.exp(logits - np.max(logits))
        priors[state] = weights / weights.sum()

    edge_stats = {}

    def expand(state):
        edge_stats[state] = {
            action: {"N": 0, "W": 0.0, "Q": 0.0, "P": float(prior)}
            for action, prior in zip(actions_by_state[state], priors[state])
        }

    expand(root_state)
    for _ in range(simulations):
        state = root_state
        path = []
        while state in legal_actions and state in edge_stats:
            total_visits = sum(edge_stats[state][action]["N"] for action in actions_by_state[state])
            root_factor = math.sqrt(total_visits)
            selected_action = actions_by_state[state][0]
            selected_score = -math.inf
            for action in actions_by_state[state]:
                statistics = edge_stats[state][action]
                score = statistics["Q"] + cpuct * statistics["P"] * root_factor / (1 + statistics["N"])
                if score > selected_score:
                    selected_action = action
                    selected_score = score
            path.append((state, selected_action))
            state = transitions[state][selected_action]

        if state in terminal_values:
            leaf_value = float(terminal_values[state])
        else:
            expand(state)
            leaf_value = float(value_predictions[state])

        leaf_player = players[state]
        for parent_state, action in reversed(path):
            backed_value = leaf_value if players[parent_state] == leaf_player else -leaf_value
            statistics = edge_stats[parent_state][action]
            statistics["N"] += 1
            statistics["W"] += backed_value
            statistics["Q"] = statistics["W"] / statistics["N"]

    root_actions = actions_by_state[root_state]
    visits = np.asarray([edge_stats[root_state][action]["N"] for action in root_actions], dtype=np.int64)
    mean_values = np.asarray([edge_stats[root_state][action]["Q"] for action in root_actions], dtype=np.float64)
    if temperature == 0:
        policy = np.zeros(len(root_actions), dtype=np.float64)
        policy[int(np.argmax(visits))] = 1.0
    else:
        log_weights = np.full(len(root_actions), -np.inf, dtype=np.float64)
        positive = visits > 0
        log_weights[positive] = np.log(visits[positive]) / temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        policy = weights / weights.sum()
    return visits, mean_values, policy