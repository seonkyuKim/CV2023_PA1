import numpy as np
from tqdm import tqdm

from const import LEFT


def warp_image(image, disparity_map, direction_biased):
    y_len = len(image)
    x_len = len(image[0])

    warped_image = np.zeros((y_len, x_len, 3), dtype=np.float64)

    for y in range(y_len):
        for x in range(x_len):
            next_x = x - disparity_map[y][x] if direction_biased == LEFT else x + disparity_map[y][x]
            if next_x < 0 or next_x >= x_len:
                continue
            warped_image[y][x] = image[y][next_x]

    return warped_image
