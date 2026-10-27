import numpy as np

def resolve_go_action(board, player, action, consecutive_passes, move_count):
    """
    Returns: the next board, player, captured count, pass count, move count, and terminal flag.
    """
    grid = np.asarray(board)
    size = grid.shape[0]
    next_move_count = move_count + 1
    move_cap = 2 * size * size

    if action == size * size:
        next_pass_count = consecutive_passes + 1
        terminal = next_pass_count >= 2 or next_move_count >= move_cap
        return grid.copy(), -player, 0, next_pass_count, next_move_count, terminal

    result = grid.copy()
    row, col = divmod(action, size)
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

    for captured_row, captured_col in captured:
        result[captured_row, captured_col] = 0

    terminal = next_move_count >= move_cap
    return result, -player, len(captured), 0, next_move_count, terminal