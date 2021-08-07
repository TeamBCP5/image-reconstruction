# 🌟카메라 이미지 품질 향상 AI 경진대회

# Contents



# Task Description

## Subject

본 대회의 주제는 빛 번짐으로 저하된 카메라 이미지 품질을 향상시키는 AI 모델 개발이었습니다. 주어진 이미지는 아래 그림과 같이 빛번짐, 블러 현상 등을 포함하여 복합적인 문제를 해결할 필요가 있었습니다.

![](https://github.com/iloveslowfood/today-I-learned/blob/main/images/image_sample_2.png?raw=true)





## Data

- 학습 데이터: 272장의 2448×3264, 350장의 1224×1632 이미지로 구성된 622장의 빛번짐 이미지
- 테스트 데이터: 20장의 2448×3264 빛번짐 이미지 



## Metric

- PSNR(Peak Signal-to-noise ratio): 신호가 가질 수 있는 최대 전력에 대한 잡음의 전력을 나타낸 것으로, 영상 또는 동영상 손실 압축에서 화질 손실 정보를 평가할때 사용됩니다.

![](https://github.com/iloveslowfood/today-I-learned/blob/main/images/psnr.png?raw=true)



# Installation

## Basic Settings

```shell
# clone repository
$ git clone https://github.com/TeamBCP5/image-reconstruction.git

# install necessary tools
$ pip install -r requirements.txt
```



## Data Structure

```shell
[dataset]/
├── train_input_img/
├── train_label_img/
└── test_input_img/
```



## Code Structure

```shell
[code]
├── configs/ # directory of model configuration files
├── data/ # modules associated with dataset
├── networks/ # modules for model architectures
├── train_modules/ # modules for train model
├── utils/ # useful utilities
├── README.md
├── requirements.txt
├── demo_augmentations.py # for testing image augmentation
├── train.py
└── inference.py
```



# Command Line Interface

## Configurations

모델 학습과 추론은 기본적으로 [모델별 Configuration 파일](https://github.com/TeamBCP5/image-reconstruction/tree/main/configs)을 바탕으로 진행됩니다. 각 Configuration 파일에는 모델 구조와 학습 데이터셋 경로 등 학습과 추론을 위한 설정값이 기록되어 있습니다. 원활한 학습/추론을 위해 데이터셋 경로 등 설정값을 환경에 맞게 설정해주세요. 주요 설정값은 다음과 같습니다.



### network: 모델 구축에 대한 설정

`name`: 모델명 설정 (`'pix2pix'`, `'hinet'`)

- \*이외 argument는 각 아키텍쳐의 arguments에 맞게 설정

- [Pix2Pix network arguments](https://github.com/TeamBCP5/image-reconstruction/blob/b3b2c3e0fe5e57215894897fa13ffd17866d2fa3/configs/Pix2Pix.yaml#L1) 
- [HINet network arguments](https://github.com/TeamBCP5/image-reconstruction/blob/b3b2c3e0fe5e57215894897fa13ffd17866d2fa3/configs/HINet_phase1.yaml#L1)



### data: 학습 시 활용할 데이터셋에 대한 설정

##### `dir`: 학습 데이터 디렉토리 경로

- `train_input_img(input 디렉토리)`, `train_label_img(label 디렉토리)` 하위 디렉토리를 포함해야 함

##### `meta`: 학습/검증 데이터 분리에 활용할 [메타 데이터](https://github.com/TeamBCP5/image-reconstruction/blob/main/configs/train_meta.csv) 경로

##### `valid_type`: 검증 전략 설정([타입별 명세 참고](https://github.com/iloveslowfood/image-reconstruction/blob/2b245bbad9421d03b943cda5402aa98ac2864c9e/data/dataset.py#L18))

##### `full_train`: 검증 데이터를 학습 데이터에 포함하여 학습을 진행할 지 여부를 설정

##### `stride (for Pix2Pix)`: sliding window 시 활용할 stride를 설정

##### `patch_size (for Pix2Pix)`: sliding window 시 각 patch의 크기를 설정

##### `denoise (for pix2pix)`: 디노이징 적용 여부 설정

##### `source (for HINet)`: HINet 모델 학습을 위한 후처리 데이터셋이 갖춰져 있지 않을 경우 메인모델(pix2pix)를 불러와 추론을 수행, 데이터를 구축하기 위한 설정

- \*후처리 데이터셋
  - *Input*. 대회에서 주어진 학습 데이터의 input 이미지에 대한 I에서 학습한 메인 모델(Pix2Pix)의 추론 결과
  - *Label*. 대회에서 주어진 학습 데이터의 label 이미지
- `config`: 후처리 데이터셋 구축에 활용할 메인모델(Pix2Pix)의 config 파일 경로
- `checkpoint`: 후처리 데이터셋 구축에 활용할 메인모델(Pix2Pix)의 pth 파일 경로



### optimizer: 학습에 활용할 optimizer 설정

##### `name`: 학습에 활용할 optimizer 이름(Adam, AdamW, ...)

##### `lr`: 학습에 활용할 initial learning rate



### scheduler: 학습에 활용할 lr scheduler 설정

##### `name`: 학습에 활용할 lr scheduler 이름(ReduceLROnPlateau, ...)

- \*이외 argument는 설정한 lr scheduler의 arguments에 맞게 설정



### checkpoint: 불러올 모델의 경로와 학습 중 모델 저장 경로

##### `load_path`: 학습을 이어 진행할 경우 불러올 모델 pth 파일 경로

##### `save_dir`: 학습 중 모델을 저장할 디렉토리 경로



## Train

최종 결과물 제출에 활용된 모델은 다음의 3단계에 걸친 학습을 통해 구축되었습니다. 

##### I. 메인 모델(Pix2Pix) 학습

- *Input*. 대회에서 주어진 학습 데이터의 input 이미지
- *Label*. 대회에서 주어진 학습 데이터의 label 이미지

##### II. 후처리 모델(HINet) 1차 학습

- *Input*. 대회에서 주어진 학습 데이터의 input 이미지
- *Label*. 대회에서 주어진 학습 데이터의 label 이미지

##### III. 후처리 모델 2차 학습

- II에서 학습한 후처리 모델(HINet)을 불러와 학습을 진행합니다.
- *Input*. 대회에서 주어진 학습 데이터의 input 이미지에 대한 I에서 학습한 메인 모델(Pix2Pix)의 추론 결과
- *Label*. 대회에서 주어진 학습 데이터의 label 이미지



#### Train All Pipeline

```shell
$ python train.py --train_type 'all'
```

- 위 학습 단계를 모두 포함한 학습을 수행합니다.

#### Train Pix2Pix in single

```shell
$ python train.py --train_type 'pix2pix'
```

- 단계 I에 해당되는 Pix2Pix 모델 학습을 수행합니다.

#### Train HINet in single

```shell
$ python train.py --train_type 'hinet'
```

- 단계 II에 해당되는 HINet 모델 학습을 수행합니다.



#### Arguments

##### `train_type`: 학습 방식 설정

- `'all'`: 위 세 단계에 걸친 학습을 진행합니다. 최종 결과물 재현에는 이 설정값을 사용됩니다.
- `pix2pix`: Pix2Pix 모델의 개별 학습을 수행합니다.
- `'hinet'`: HINet 모델의 개별 학습을 수행합니다. '단계 II. 후처리 모델(HINet) 1차 학습'을 기준으로 학습이 진행됩니다.

##### `config_pix2pix`: Pix2Pix 모델 configuration 파일 경로

##### `config_hinet_phase1`: HINet 모델(phase1) configuration 파일 경로

##### `config_hinet_phase2`: HINet 모델(phase2) configuration 파일 경로



## Inference

```shell
$ python inference.py --checkpoint_main "./checkpoints/pix2pix/pix2pix.pth" --checkpoint_post "./checkpoints/hinet/hinet.pth" --image_dir "/content/data/test_input_img"
```

메인 모델(Pix2Pix)과 후처리 모델(HINet)을 불러와 추론을 수행합니다. 추론은 다음의 두 단계를 거쳐 진행됩니다.

##### I. 메인 모델(Pix2Pix) 추론

- Input: 대회에서 주어진 테스트 데이터의 input 이미지

##### II. 후처리 모델(HINet) 1차 학습

- Input: 단계 I에서 메인 모델의 추론 결과
- 해당 단계에서의 결과물이 최종 추론 결과물로 저장됩니다.



#### Arguments

##### `config_main`: Main 모델(Pix2Pix) config 파일 경로

##### `config_post`: Postprocessing 모델(HINet) config 파일 경로

##### `checkpoint_main`: 학습한 main 모델(Pix2Pix)의 pth 파일 경로

##### `checkpoint_post`: 학습한 postprocessing 모델(HINet)의 pth 파일 경로

##### `image_dir`: 추론 시 사용될 데이터 디렉토리 경로

##### `patch_size`: 추론 시 사용될 이미지 patch의 크기

##### `stride`: 추론 시 사용될 stride의 크기

##### `batch_size`: 추론 시 사용될 batch의 크기

##### `output_dir`: 추론 결과를 저장할 디렉토리 경로. 해당 디렉토리 내 압축파일 형태로 결과물이 저장됩니다.

