# Configurations

모델 학습과 추론은 기본적으로 [모델별 Configuration 파일](https://github.com/TeamBCP5/image-reconstruction/tree/main/configs)을 바탕으로 진행됩니다. 각 Configuration 파일에는 모델 구조와 학습 데이터셋 경로 등 학습과 추론을 위한 설정값이 기록되어 있습니다. 원활한 학습/추론을 위해 데이터셋 경로 등 설정값을 환경에 맞게 설정해주세요. 주요 설정값은 다음과 같습니다.



### network. 모델 구축에 대한 설정

`name`: 모델명 설정 (`'pix2pix'`, `'hinet'`)

- 이외 argument는 각 아키텍쳐의 arguments에 맞게 설정

- [Pix2Pix network arguments](https://github.com/TeamBCP5/image-reconstruction/blob/b3b2c3e0fe5e57215894897fa13ffd17866d2fa3/configs/Pix2Pix.yaml#L1) 
- [HINet network arguments](https://github.com/TeamBCP5/image-reconstruction/blob/b3b2c3e0fe5e57215894897fa13ffd17866d2fa3/configs/HINet_phase1.yaml#L1)



### data. 학습 시 활용할 데이터셋에 대한 설정

`dir`: 학습 데이터 디렉토리 경로

- `train_input_img(input 디렉토리)`, `train_label_img(label 디렉토리)` 하위 디렉토리를 포함해야 함

`meta`: 학습/검증 데이터 분리에 활용할 [메타 데이터](https://github.com/TeamBCP5/image-reconstruction/blob/main/configs/train_meta.csv) 경로

`valid_type`: 검증 전략 설정([타입별 명세 참고](https://github.com/iloveslowfood/image-reconstruction/blob/2b245bbad9421d03b943cda5402aa98ac2864c9e/data/dataset.py#L18))

`full_train`: 검증 데이터를 학습 데이터에 포함하여 학습을 진행할 지 여부를 설정

`stride (for Pix2Pix)`: sliding window 시 활용할 stride를 설정

`patch_size (for Pix2Pix)`: sliding window 시 각 patch의 크기를 설정

`denoise (for pix2pix)`: 디노이징 적용 여부 설정

`source (for HINet)`: HINet 모델 학습을 위한 데이터셋이 갖춰져 있지 않을 경우 메인모델(pix2pix)를 불러와 추론을 수행, 데이터를 구축하기 위한 설정

- 후처리 데이터셋?
  - ***Input***. 대회에서 주어진 학습 데이터의 input 이미지에 대한 I에서 학습한 메인 모델(Pix2Pix)의 추론 결과
  - ***Label***. 대회에서 주어진 학습 데이터의 label 이미지
- `config`: 후처리 데이터셋 구축에 활용할 메인모델(Pix2Pix)의 config 파일 경로
- `checkpoint`: 후처리 데이터셋 구축에 활용할 메인모델(Pix2Pix)의 pth 파일 경로



### optimizer. 학습에 활용할 optimizer 설정

`name`: 학습에 활용할 optimizer 이름(Adam, AdamW, ...)

`lr`: 학습에 활용할 initial learning rate



### scheduler. 학습에 활용할 lr scheduler 설정

`name`: 학습에 활용할 learning rate scheduler 이름(ReduceLROnPlateau, ...)

- 이외 argument는 설정한 learning rate scheduler의 arguments에 맞게 설정



### checkpoint. 불러올 모델의 경로와 학습 중 모델 저장 경로

`load_path`: 학습을 이어 진행할 경우 불러올 모델 pth 파일 경로

`save_dir`: 학습 중 모델을 저장할 디렉토리 경로