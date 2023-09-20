import math

import cv2
import numpy as np
from tqdm import tqdm
import os

from Birchfield_Tomasi_dissimilarity import Birchfield_Tomasi_dissimilarity
from aggregate_cost_volume import aggregate_cost_volume
from const import LEFT, RIGHT, PATCH_SIZE
from warp import warp_image


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d, direction_biased, index):
    cost_volume_file_name = f'output/Cost/cost_volume_{PATCH_SIZE}_{index}.npy'
    cost_disparity_file_name = f'output/Cost/cost_disparity_{PATCH_SIZE}_{index}.npy'
    cost_disparity_image_name = f'output/Cost/cost_disparity_{PATCH_SIZE}_{index}.png'

    # 캐시처럼 사용. 존재하지 않을 경우 저장
    if not os.path.exists(cost_volume_file_name) or not os.path.exists(cost_disparity_file_name):
        print(f'cost_volume_{index}.npy does not exist. Generating...')
        cost_volume, cost_disparity = Birchfield_Tomasi_dissimilarity(left_image, right_image, d, direction_biased)

        with open(cost_volume_file_name, 'wb') as f:
            np.save(f, cost_volume)

        with open(cost_disparity_file_name, 'wb') as f:
            np.save(f, cost_disparity)

    # 존재하면 불러와서 사용.
    with open(cost_volume_file_name, 'rb') as f:
        cost_volume = np.load(f)

    with open(cost_disparity_file_name, 'rb') as f:
        cost_disparity = np.load(f)

    # 중간 결과 저장
    result_image = np.zeros((len(cost_disparity), len(cost_disparity[0])))
    for y in range(len(cost_disparity)):
        for x in range(len(cost_disparity[0])):
            result_image[y][x] = cost_disparity[y][x] / d * 255

    cv2.imwrite(cost_disparity_image_name, result_image)

    aggregated_volume_file_name = f'output/Final_Disparity/aggregated_cost_volume_{PATCH_SIZE}_{index}.npy'
    aggregated_disparity_file_name = f'output/Final_Disparity/aggregated_disparity_{PATCH_SIZE}_{index}.npy'
    aggregated_disparity_image_name = f'output/Final_Disparity/aggregated_disparity_{PATCH_SIZE}_{index}.png'

    # 캐시처럼 사용. 존재하지 않을 경우 저장
    if not os.path.exists(aggregated_volume_file_name) or not os.path.exists(aggregated_disparity_file_name):
        print(f'aggregated_cost_volume_{index}.npy does not exist. Generating...')
        aggregated_cost_volume = aggregate_cost_volume(cost_volume)
        aggregated_disparity = aggregated_cost_volume.argmin(axis=2)

        with open(aggregated_volume_file_name, 'wb') as f:
            np.save(f, aggregated_cost_volume)

        with open(aggregated_disparity_file_name, 'wb') as f:
            np.save(f, aggregated_disparity)

    # 존재하면 불러와서 사용.
    with open(aggregated_volume_file_name, 'rb') as f:
        aggregated_cost_volume = np.load(f)

    with open(aggregated_disparity_file_name, 'rb') as f:
        aggregated_disparity = np.load(f)

    semi_global_image = np.zeros((len(aggregated_cost_volume), len(aggregated_cost_volume[0])))
    for y in range(len(aggregated_cost_volume)):
        for x in range(len(aggregated_cost_volume[0])):
            semi_global_image[y][x] = aggregated_disparity[y][x] / d * 255

    cv2.imwrite(aggregated_disparity_image_name, semi_global_image)

    return aggregated_disparity, direction_biased


if __name__ == "__main__":
    img_list = list()
    ground_truth = None
    noise = 25

    # Load required images
    target_image = None

    for i in range(1, 8):
        img = cv2.imread(f"input/0{i}_noise{noise}.png")  # 215 x 328 image
        img = img.astype(np.float64)

        if i == 4:
            target_image = img
        else:
            img_list.append(img)

    d = 24

    disparity_list = list()

    for i in range(len(img_list)):
        # 4번 이미지 기준으로 왼쪽으로 치우쳐져 있는지, 오른쪽으로 치우쳐져 있는지
        direction_biased = RIGHT if i < 3 else LEFT
        aggregated_disparity, direction_biased = semi_global_matching(target_image, img_list[i], d, direction_biased, i)
        disparity_list.append((aggregated_disparity, direction_biased))

    warped_image_list = list()
    for i, image in enumerate(img_list):
        aggregated_disparity, direction_biased = disparity_list[i]
        warped_image = warp_image(image, aggregated_disparity, direction_biased)
        cv2.imwrite(f'output/Warped/warped_image_{PATCH_SIZE}_{i}.png', warped_image)
        warped_image_list.append(warped_image)

    boundary_range = d
    ground_truth = cv2.imread(f"target/gt.png")
    ground_truth = ground_truth.astype(np.float64)
    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]

    # Aggregate warped images
    aggregated_warped_image = np.sum(warped_image_list, axis=0) / len(warped_image_list)
    cv2.imwrite(f'output/Warped/aggregated_warped_image_{PATCH_SIZE}.png', aggregated_warped_image)
    cropped_aggregated_warped_image = aggregated_warped_image[boundary_range:-boundary_range,
                                      boundary_range:-boundary_range]

    # Compute MSE and PSNR
    before_mse = 0
    cropped_target_image = target_image[boundary_range:-boundary_range, boundary_range:-boundary_range]
    for y in range(len(cropped_ground_truth)):
        for x in range(len(cropped_ground_truth[0])):
            before_mse += np.mean((cropped_ground_truth[y][x] - cropped_target_image[y][x]) ** 2)

    before_mse = before_mse / (len(cropped_ground_truth) * len(cropped_ground_truth[0]))

    before_psnr = 20 * math.log10(255) - 10 * math.log10(before_mse)

    # denoising 후 MSE, PSNR
    mse = 0
    for y in range(len(cropped_ground_truth)):
        for x in range(len(cropped_ground_truth[0])):
            mse += np.mean((cropped_ground_truth[y][x] - cropped_aggregated_warped_image[y][x]) ** 2)

    mse = mse / (len(cropped_ground_truth) * len(cropped_ground_truth[0]))
    psnr = 20 * math.log10(255) - 10 * math.log10(mse)

    text = f"""noise: {noise}
patch size: {PATCH_SIZE}
[BEFORE]
MSE: {before_mse}
PSNR: {before_psnr}
[AFTER]
MSE: {mse}
PSNR: {psnr}
"""

    print(text)

    with open(f'output/result_{PATCH_SIZE}.txt', 'w') as f:
        f.write(text)
