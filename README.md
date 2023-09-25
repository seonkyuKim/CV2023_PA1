# Install

```bash
pip3 install -r requirements.txt
```

# Usage

```bash
python3 Semi_Global_Matching.py 
```

`ouput` 디렉토리 안에 .npy 데이터를 저장해두어 캐시로 사용합니다. 만약 해당 파일이 존재할 경우 계산이 오래 걸리는 cost_volume 등을 다시 구하지 않습니다.
캐시 파일들은 용량이 너무 큰 관계로 git 에 commit 하지 않았습니다.

MSE, PSNR 결과는 콘솔에 출력되고, output/Noise_{NOISE}/result_{PATCH_SIZE}.txt 에 결과 이미지가 저장됩니다.

- const.py 에서 PATCH_SIZE 변수를 조정할 수 있습니다.
- const.py 에서 NOISE 변수를 조정할 수 있습니다.

## output 디렉토리 파일 설명

ground truth 는 input 의 7개 이미지 중 4번째 이미지입니다.
아래에서 사용되는 index 는 7개 중 4번째 이미지가 제거된 index 입니다.
따라서 index 0, 1, 2 는 1, 2, 3 번째 이미지를 의미하고 index 3, 4, 5 는 5, 6, 7 번째 이미지를 의미합니다.

최종 결과는 result_{PATCH_SIZE}.txt 에 저장됩니다.

### Cost 디렉토리

cost volume 계산 값과 중간 disparity map 을 저장합니다.

- cost_disparity_{PATCH_SIZE}_{index}.png: 중간 disparity map
- cost_disparity_{PATCH_SIZE}_{index}.npy: 중간 disparity map numpy 캐시 파일
- cost_volume_{PATCH_SIZE}_{index}.npy: cost volume numpy 캐시 파일

### Final_Disparity 디렉토리

최종 semi global matching 을 한 이후 결과를 저장합니다.

- aggregated_disparity_{PATCH_SIZE}_{index}.png: aggregated disparity map
- aggregated_disparity_{PATCH_SIZE}_{index}.npy: aggregated disparity map numpy 캐시 파일
- aggregated_cost_volume_{PATCH_SIZE}_{index}.npy: aggregated cost volume numpy 캐시 파일

### Warped 디렉토리

와핑된 이미지들과 이들을 aggregate 한 이미지를 저장합니다.

- warped_image_{PATCH_SIZE}_{index}.png: 와핑된 이미지
- aggregated_warped_image_{PATCH_SIZE}.png: 와핑된 이미지를 aggregate 한 이미지

# Github

코드는 github 에서도 확인할 수 있습니다: https://github.com/seonkyuKim/CV2023_PA1