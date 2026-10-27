import numpy as np

def tromp_taylor_result(board, komi, consecutive_passes, move_count):
    """
    Returns: terminal status, black score, white score, and the winning colour.
    """
    grid = np.asarray(board)
    size = grid.shape[0]
    if consecutive_passes < 2 and move_count < 2 * size * size:
        return False, None, None, None

    black_score = np.float64(np.count_nonzero(grid == 1))
    white_score = np.float64(np.count_nonzero(grid == -1) + komi)
    visited = set()

    for row in range(size):
        for col in range(size):
            if grid[row, col] != 0 or (row, col) in visited:
                continue
            stack = [(row, col)]
            region = set()
            boundary = set()
            while stack:
                point = stack.pop()
                if point in region:
                    continue
                region.add(point)
                visited.add(point)
                r, c = point
                for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                    if not (0 <= nr < size and 0 <= nc < size):
                        continue
                    if grid[nr, nc] == 0:
                        if (nr, nc) not in region:
                            stack.append((nr, nc))
                    else:
                        boundary.add(int(grid[nr, nc]))

            if boundary == {1}:
                black_score += len(region)
            elif boundary == {-1}:
                white_score += len(region)

    winner = 1 if black_score > white_score else -1 if white_score > black_score else 0
    return True, black_score, white_score, winner