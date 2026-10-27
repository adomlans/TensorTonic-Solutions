import copy

def sample_recent_replay_batch(games, window_size, position_indices):
    recent_games = games[-min(window_size, len(games)):]
    eligible = [record for game in recent_games for record in game]
    sampled_records = [copy.deepcopy(eligible[index]) for index in position_indices]
    return sampled_records, len(eligible)