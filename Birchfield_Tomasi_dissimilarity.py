import numpy as np
from tqdm import tqdm

from const import LEFT, MARGIN
from intensity import get_intensity


def get_cost_volume(left_image, right_image, d, direction_biased):
    y_len = len(left_image)
    x_len = len(left_image[0])

    cost_volume = np.full((y_len, x_len, d), np.inf)

    # 왼쪽으로 치우쳐져 있다면, left cost volume 을 만들어야 함
    if direction_biased == LEFT:
        # left cost volume
        for disparity in tqdm(range(d)):
            # disparity_map = np.full((y_len, x_len), np.inf)
            for y in range(y_len):
                for x in range(disparity, x_len):
                    cost_volume[y][x][disparity] = SAD(get_patch(left_image, y, x),
                                                       get_patch(right_image, y, x - disparity))

    # 오른쪽으로 치우쳐져 있다면, right cost volume 을 만들어야 함
    else:
        # right cost volume
        for disparity in tqdm(range(d)):
            for y in range(y_len):
                for x in range(x_len - disparity):
                    cost_volume[y][x][disparity] = SAD(get_patch(left_image, y, x),
                                                       get_patch(right_image, y, x + disparity))

    cost_disparity = cost_volume.argmin(axis=2)

    return cost_volume, cost_disparity


def get_patch(image, y, x):
    # input: 이미지, y좌표, x좌표, 패치 크기
    # output: 패치 크기만큼의 픽셀들의 1차원 배열
    y_len = len(image)
    x_len = len(image[0])
    patch = list()
    for i in range(y - MARGIN, y + MARGIN + 1):
        for j in range(x - MARGIN, x + MARGIN + 1):
            if i < 0 or i >= y_len or j < 0 or j >= x_len:
                continue
            patch.append(image[i][j])

    return patch


def SAD(left_pixels, right_pixels):
    # input: RGB 픽셀들의 1차원 배열
    sum_value = 0
    for l, r in zip(left_pixels, right_pixels):
        sum_value += abs(get_intensity(l) - get_intensity(r))

    return sum_value / len(left_pixels)
