import numpy as np

def legal_go_action_mask(board, player, seen_positions, consecutive_passes, move_count):
    """
    Returns: a Boolean vector containing one legality flag per board action and pass.
    """
    grid = np.asarray(board)
    size = grid.shape[0]
    mask = np.zeros(size * size + 1, dtype=bool)
    if consecutive_passes >= 2 or move_count >= 2 * size * size:
        return mask

    def apply(action):
        row, col = divmod(action, size)
        if grid[row, col] != 0:
            return None
        result = grid.copy()
        result[row, col] = player

        def group_and_liberties(start_row, start_col):
            colour = result[start_row, start_col]
            stack = [(start_row, start_col)]
            group = set()
            liberties = set()
            while stack:
                point = stack.pop()
                if point in group:
                    continue
                group.add(point)
                r, c = point
                for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                    if not (0 <= nr < size and 0 <= nc < size):
                        continue
                    if result[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif result[nr, nc] == colour and (nr, nc) not in group:
                        stack.append((nr, nc))
            return group, liberties

        captured = set()
        for nr, nc in ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)):
            if 0 <= nr < size and 0 <= nc < size and result[nr, nc] == -player:
                group, liberties = group_and_liberties(nr, nc)
                if not liberties:
                    captured.update(group)
        for r, c in captured:
            result[r, c] = 0

        _, liberties = group_and_liberties(row, col)
        if not liberties:
            return None
        if any(np.array_equal(result, prior) for prior in seen_positions):
            return None
        return result

    for action in range(size * size):
        mask[action] = apply(action) is not None
    mask[-1] = True
    return mask