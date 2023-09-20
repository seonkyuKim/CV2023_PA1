import numpy as np
from tqdm import tqdm

P1 = 5
P2 = 150


def aggregate_cost_volume(cost_volume):
    forward_pass = [(0, 1), (1, 0), (1, 1), (1, -1)]
    backward_pass = [(0, -1), (-1, 0), (-1, -1), (-1, 1)]

    # aggregated_costs 는 y, x, d, r 의 4차원 배열
    y_len = len(cost_volume)
    x_len = len(cost_volume[0])
    d_len = len(cost_volume[0][0])
    aggregated_costs = np.full(
        (y_len, x_len, d_len, len(forward_pass + backward_pass)), np.inf)

    for idx, (dy, dx) in enumerate(forward_pass):
        for y in tqdm(range(y_len)):
            for x in range(x_len):
                for d in range(d_len):
                    before_y = y - dy
                    before_x = x - dx
                    before_d = d - 1
                    after_d = d + 1

                    first_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len else \
                        aggregated_costs[before_y][before_x][d][idx]

                    second_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or before_d < 0 else \
                        aggregated_costs[before_y][before_x][d - 1][idx] + P1

                    third_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or after_d >= d_len else \
                        aggregated_costs[before_y][before_x][d + 1][idx] + P1

                    forth_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or after_d < 0 or after_d >= d_len else \
                        np.min(aggregated_costs[before_y][before_x][after_d:, idx] + P2)

                    minus_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len else \
                        np.min(cost_volume[before_y][before_x])

                    min_term = min(first_term, second_term, third_term, forth_term)
                    if min_term is np.inf and minus_term is np.inf:
                        aggregated_costs[y][x][d][idx] = cost_volume[y][x][d]
                    else:
                        aggregated_costs[y][x][d][idx] = cost_volume[y][x][d] + min_term - minus_term

    for idx, (dy, dx) in enumerate(backward_pass):
        for y in tqdm(reversed(range(y_len))):
            for x in reversed(range(x_len)):
                for d in range(d_len):
                    before_y = y - dy
                    before_x = x - dx
                    before_d = d - 1
                    after_d = d + 1

                    first_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len else \
                        aggregated_costs[before_y][before_x][d][idx + 4]

                    second_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or before_d < 0 else \
                        aggregated_costs[before_y][before_x][d - 1][idx + 4] + P1

                    third_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or after_d >= d_len else \
                        aggregated_costs[before_y][before_x][d + 1][idx + 4] + P1

                    forth_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len or after_d < 0 or after_d >= d_len else \
                        np.min(aggregated_costs[before_y][before_x][after_d:, idx + 4] + P2)

                    minus_term = np.inf \
                        if before_y < 0 or before_y >= y_len or before_x < 0 or before_x >= x_len else \
                        np.min(cost_volume[before_y][before_x])

                    min_term = min(first_term, second_term, third_term, forth_term)
                    if min_term is np.inf and minus_term is np.inf:
                        aggregated_costs[y][x][d][idx + 4] = cost_volume[y][x][d]
                    else:
                        aggregated_costs[y][x][d][idx + 4] = cost_volume[y][x][d] + min_term - minus_term

    # aggregated_costs 는 y, x, d, r 의 4차원 배열
    aggregated_volume = np.sum(aggregated_costs, axis=3)
    return aggregated_volume
