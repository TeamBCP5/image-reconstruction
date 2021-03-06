# ๐์นด๋ฉ๋ผ ์ด๋ฏธ์ง ํ์ง ํฅ์ AI ๊ฒฝ์ง๋ํ

# Contents

#### **[๐ง Task Description](https://github.com/TeamBCP5/image-reconstruction#task-description)**

#### **[โ Installation](https://github.com/TeamBCP5/image-reconstruction#installation)**

#### **[๐น Command Line Interface](https://github.com/TeamBCP5/image-reconstruction#command-line-interface)**

- [**Configurations**](https://github.com/TeamBCP5/image-reconstruction#configurations)  
- [**Train**](https://github.com/TeamBCP5/image-reconstruction#train)  
- [**Inference**](https://github.com/TeamBCP5/image-reconstruction#inference)  
- [**Demo Augmentation**](https://github.com/TeamBCP5/image-reconstruction#demo-augmentation)  
- [**Demo Preprocessing**](https://github.com/TeamBCP5/image-reconstruction#demo-preprocessing)  

---



# ๐งTask Description

## Subject

๋ณธ ๋ํ์ ์ฃผ์ ๋ ๋น ๋ฒ์ง์ผ๋ก ์ ํ๋ ์นด๋ฉ๋ผ ์ด๋ฏธ์ง ํ์ง์ ํฅ์์ํค๋ AI ๋ชจ๋ธ ๊ฐ๋ฐ์ด์์ต๋๋ค. ์ฃผ์ด์ง ์ด๋ฏธ์ง๋ ์๋ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด ๋น๋ฒ์ง, ๋ธ๋ฌ ํ์ ๋ฑ์ ํฌํจํ์ฌ ๋ณตํฉ์ ์ธ ๋ฌธ์ ๋ฅผ ํด๊ฒฐํ  ํ์๊ฐ ์์์ต๋๋ค.

![](https://github.com/iloveslowfood/today-I-learned/blob/main/images/image_sample_2.png?raw=true)





## Data

- ํ์ต ๋ฐ์ดํฐ: 272์ฅ์ 2448ร3264, 350์ฅ์ 1224ร1632 ์ด๋ฏธ์ง๋ก ๊ตฌ์ฑ๋ 622์ฅ์ ๋น๋ฒ์ง ์ด๋ฏธ์ง
- ํ์คํธ ๋ฐ์ดํฐ: 20์ฅ์ 2448ร3264 ๋น๋ฒ์ง ์ด๋ฏธ์ง 



## Metric

- PSNR(Peak Signal-to-noise ratio): ์ ํธ๊ฐ ๊ฐ์ง ์ ์๋ ์ต๋ ์ ๋ ฅ์ ๋ํ ์ก์์ ์ ๋ ฅ์ ๋ํ๋ธ ๊ฒ์ผ๋ก, ์์ ๋๋ ๋์์ ์์ค ์์ถ์์ ํ์ง ์์ค ์ ๋ณด๋ฅผ ํ๊ฐํ ๋ ์ฌ์ฉ๋ฉ๋๋ค.

![](https://github.com/iloveslowfood/today-I-learned/blob/main/images/psnr.png?raw=true)



---



# โInstallation

## Preparation

```shell
# clone repository
$ git clone https://github.com/TeamBCP5/image-reconstruction.git

# install necessary tools
$ pip install -r requirements.txt
```



## Data Structure

```shell
# Download: https://dacon.io/competitions/official/235746/data
[camera_dataset]/
โโโ train_input_img/ # ํ์ต ๋ฐ์ดํฐ ์๋ ฅ ์ด๋ฏธ์ง
โโโ train_label_img/
โโโ hinet_dataset/ # postprocessing ๋ฐ์ดํฐ์ ๋๋ ํ ๋ฆฌ NOTE. ํ์ต ๊ณผ์  ์ค ๊ตฌ์ถ๋๋ ๋๋ ํ ๋ฆฌ์๋๋ค.
โ     โโโ train_input_img/
โ     โโโ train_label_img/
โโโ test_input_img/
```



## Code Structure

```shell
[code]
โโโ camera_dataset/ # ๋ฐ์ดํฐ์ ๋๋ ํ ๋ฆฌ
โโโ configs/ # ๋ชจ๋ธ config ํ์ผ ๋๋ ํ ๋ฆฌ
โโโ data/ # data ์ฒ๋ฆฌ ๊ด๋ จ ๋ชจ๋ ๋๋ ํ ๋ฆฌ
โโโ networks/ # ๋ชจ๋ธ ์ํคํ์ฒ ๊ด๋ จ ๋ชจ๋ ๋๋ ํ ๋ฆฌ
โโโ train_modules/ # ๋ชจ๋ธ ํ์ต ๊ด๋ จ ๋ชจ๋ ๋๋ ํ ๋ฆฌ
โโโ utils/ # ์ ํธ๋ฆฌํฐ ๊ด๋ จ ๋ชจ๋ ๋๋ ํ ๋ฆฌ
โโโ README.md
โโโ requirements.txt
โโโ demo_augmentations.py # Augmentation ํ์คํธ๋ฅผ ์ํ ์คํฌ๋ฆฝํธ ํ์ผ
โโโ demo_preprocessing.py # Preprocessing ํ์คํธ๋ฅผ ์ํ ์คํฌ๋ฆฝํธ ํ์ผ
โโโ train.py
โโโ inference.py
```



---



# ๐นCommand Line Interface

์นด๋ฉ๋ผ ์ด๋ฏธ์ง ํ์ง ๊ฐ์  ๊ณผ์ ์ ๋๋ต ๋ค์ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ต๋๋ค. Sliding Window ๊ธฐ๋ฐ์ Pix2Pix ๋ชจ๋ธ์ ํตํด 1์ฐจ์ ์ผ๋ก ๋น๋ฒ์ง์ ์ ๊ฑฐํ ๋ค, HINet ๋ชจ๋ธ์ ํตํด ๊ฒฉ์ ๋ฌด๋ฌ ๋ฑ ์์๋ ํ์ง์ ๋ณด์ํฉ๋๋ค. ํนํ, ํ์ต ๋จ๊ณ์์ Pix2Pix Generator๋ Discriminator์ ํจ๊ป ํ์ต๋ฉ๋๋ค.

![](https://github.com/iloveslowfood/today-I-learned/blob/main/images/pipeline.png?raw=true)



## Configurations

๋ชจ๋ธ ํ์ต๊ณผ ์ถ๋ก ์ ๊ธฐ๋ณธ์ ์ผ๋ก [**๋ชจ๋ธ๋ณ Configuration ํ์ผ**](https://github.com/TeamBCP5/image-reconstruction/tree/main/configs)์ ๋ฐํ์ผ๋ก ์งํ๋ฉ๋๋ค. ๊ฐ Configuration ํ์ผ์๋ ๋ชจ๋ธ ๊ตฌ์กฐ์ ํ์ต ๋ฐ์ดํฐ์ ๊ฒฝ๋ก ๋ฑ ํ์ต๊ณผ ์ถ๋ก ์ ์ํ ์ค์ ๊ฐ์ด ๊ธฐ๋ก๋์ด ์์ต๋๋ค. ์ํํ ํ์ต/์ถ๋ก ์ ์ํด์๋ ๋ฐ์ดํฐ์ ๊ฒฝ๋ก ๋ฑ ์ค์ ๊ฐ์ ํ๊ฒฝ์ ๋ง๊ฒ ์ค์ ํด์ฃผ์์ผ ํฉ๋๋ค. Configuration ํ์ผ ๋ช์ธ๋ [**์ด๊ณณ**](https://github.com/TeamBCP5/image-reconstruction/blob/main/Configurations.md)์์ ํ์ธํ์ค ์ ์์ต๋๋ค.



## Train

์ต์ข ๊ฒฐ๊ณผ๋ฌผ ์ ์ถ์ ํ์ฉ๋ ๋ชจ๋ธ์ ๋ค์์ 3๋จ๊ณ์ ๊ฑธ์น ํ์ต์ ํตํด ์ ์๋์์ต๋๋ค. 

#### I. ๋ฉ์ธ ๋ชจ๋ธ(Pix2Pix) ํ์ต

- Sliding Window ๋ฐฉ๋ฒ์ ๋ฐํ์ผ๋ก ์ด๋ฏธ์ง ํ์ง์ ํฅ์์ํค๋ ๋ฉ์ธ ๋ชจ๋ธ(Pix2Pix)์ ํ์ตํฉ๋๋ค.
- ***Input***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ input ์ด๋ฏธ์ง
- ***Label***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ label ์ด๋ฏธ์ง

#### II. ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet) 1์ฐจ ํ์ต

- ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet)์ ์ฃผ์ด์ง ๋ฐ์ดํฐ๋ฅผ ํ์ฉํ์ฌ 1์ฐจ์ ์ผ๋ก ํ์ตํฉ๋๋ค.
- ***Input***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ input ์ด๋ฏธ์ง
- ***Label***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ label ์ด๋ฏธ์ง

#### III. ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet) 2์ฐจ ํ์ต

- II์์ ํ์ตํ ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet)์ ๋ถ๋ฌ์ ํ์ต์ ์งํํฉ๋๋ค.
- ***Input***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ input ์ด๋ฏธ์ง์ ๋ํ I์์ ํ์ตํ ๋ฉ์ธ ๋ชจ๋ธ(Pix2Pix)์ ์ถ๋ก  ๊ฒฐ๊ณผ
- ***Label***. ๋ํ์์ ์ฃผ์ด์ง ํ์ต ๋ฐ์ดํฐ์ label ์ด๋ฏธ์ง



### Train All Pipeline

์ ํ์ต ๋จ๊ณ๋ฅผ ๋ชจ๋ ํฌํจํ ํ์ต์ ์ํํฉ๋๋ค.

```shell
$ python train.py --train_type 'all'
```

### Train Pix2Pix in single

๋จ๊ณ I์ ํด๋น๋๋ Pix2Pix ๋ชจ๋ธ ํ์ต์ ์ํํฉ๋๋ค.

```shell
$ python train.py --train_type 'pix2pix'
```

### Train HINet in single

๋จ๊ณ II์ ํด๋น๋๋ HINet ๋ชจ๋ธ ํ์ต์ ์ํํฉ๋๋ค.

```shell
$ python train.py --train_type 'hinet'
```



### Arguments

`train_type`: ํ์ต ๋ฐฉ์ ์ค์ 

- `'all'`: ์ ์ธ ๋จ๊ณ์ ๊ฑธ์น ํ์ต์ ์งํํฉ๋๋ค. ์ต์ข ๊ฒฐ๊ณผ๋ฌผ ์ฌํ์๋ ์ด ์ค์ ๊ฐ์ ์ฌ์ฉ๋ฉ๋๋ค.
- `'pix2pix'`: Pix2Pix ๋ชจ๋ธ์ ๊ฐ๋ณ ํ์ต์ ์ํํฉ๋๋ค.
- `'hinet'`: HINet ๋ชจ๋ธ์ ๊ฐ๋ณ ํ์ต์ ์ํํฉ๋๋ค. '๋จ๊ณ II. ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet) 1์ฐจ ํ์ต'์ ๊ธฐ์ค์ผ๋ก ํ์ต์ด ์งํ๋ฉ๋๋ค.

`config_pix2pix`: Pix2Pix ๋ชจ๋ธ configuration ํ์ผ ๊ฒฝ๋ก

`config_hinet_phase1`: HINet ๋ชจ๋ธ(phase1) configuration ํ์ผ ๊ฒฝ๋ก

`config_hinet_phase2`: HINet ๋ชจ๋ธ(phase2) configuration ํ์ผ ๊ฒฝ๋ก



## Inference

๋ฉ์ธ ๋ชจ๋ธ(Pix2Pix)๊ณผ ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet)์ ๋ถ๋ฌ์ ์ถ๋ก ์ ์ํํฉ๋๋ค. ์ถ๋ก ์ ๋ค์์ ๋ ๋จ๊ณ๋ฅผ ๊ฑฐ์ณ ์งํ๋ฉ๋๋ค.

```shell
$ python inference.py --checkpoint_main "./best_models/pix2pix.pth" --checkpoint_post "./best_models/hinet.pth" --image_dir "./camera_dataset/test_input_img"
```

#### I. ๋ฉ์ธ ๋ชจ๋ธ(Pix2Pix) ์ถ๋ก 

- ***Input***. ๋ํ์์ ์ฃผ์ด์ง ํ์คํธ ๋ฐ์ดํฐ์ input ์ด๋ฏธ์ง

#### II. ํ์ฒ๋ฆฌ ๋ชจ๋ธ(HINet) 1์ฐจ ํ์ต

- ***Input***. ๋จ๊ณ I์์ ๋ฉ์ธ ๋ชจ๋ธ์ ์ถ๋ก  ๊ฒฐ๊ณผ
- ํด๋น ๋จ๊ณ์์์ ๊ฒฐ๊ณผ๋ฌผ์ด ์ต์ข ์ถ๋ก  ๊ฒฐ๊ณผ๋ฌผ๋ก ์ ์ฅ๋ฉ๋๋ค.



### Arguments

`config_main`: Main ๋ชจ๋ธ(Pix2Pix) config ํ์ผ ๊ฒฝ๋ก

`config_post`: Postprocessing ๋ชจ๋ธ(HINet) config ํ์ผ ๊ฒฝ๋ก

`checkpoint_main`: ํ์ตํ main ๋ชจ๋ธ(Pix2Pix)์ pth ํ์ผ ๊ฒฝ๋ก

`checkpoint_post`: ํ์ตํ postprocessing ๋ชจ๋ธ(HINet)์ pth ํ์ผ ๊ฒฝ๋ก

`image_dir`: ์ถ๋ก  ์ ์ฌ์ฉ๋  ๋ฐ์ดํฐ ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก

`patch_size`: ์ถ๋ก  ์ ์ฌ์ฉ๋  ์ด๋ฏธ์ง patch์ ํฌ๊ธฐ

`stride`: ์ถ๋ก  ์ ์ฌ์ฉ๋  stride์ ํฌ๊ธฐ

`batch_size`: ์ถ๋ก  ์ ์ฌ์ฉ๋  batch์ ํฌ๊ธฐ

`output_dir`: ์ถ๋ก  ๊ฒฐ๊ณผ๋ฅผ ์ ์ฅํ  ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก. ํด๋น ๋๋ ํ ๋ฆฌ ๋ด ์์ถํ์ผ ํํ๋ก ๊ฒฐ๊ณผ๋ฌผ์ด ์ ์ฅ๋ฉ๋๋ค.



## Demo Augmentation

๋ชจ๋ธ ํ์ต์ ํ์ฉํ data augmentation์ ์์ ๊ฒฐ๊ณผ๋ฌผ์ ์์ฑํฉ๋๋ค.

```shell
$ python demo_augmentation.py --data_dir "./camera_dataset/" --num_samples 10 --save_dir './sample_augmentation/'
```



### Outputs

```shell
[SAVE_DIR]
โโโ original/ # ์๋ณธ ์ด๋ฏธ์ง
โโโ hinet/ # HINet์ ์ํ data augmentation ๊ฒฐ๊ณผ๋ฌผ
โโโ pix2pix/ # pix2pix๋ฅผ ์ํ data augmentation ๊ฒฐ๊ณผ๋ฌผ
```

### Arguments

`data_dir`: input ๋ฐ์ดํฐ ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก

`num_samples`: ์์ฑํ  ์ํ ์

`save_dir`: Augmentation ์ ์ฉ ๊ฒฐ๊ณผ๋ฅผ ์ ์ฅํ  ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก



## Demo Preprocessing

๋ชจ๋ธ ํ์ต์ ํ์ฉํ data preprocessing์ ์์ ๊ฒฐ๊ณผ๋ฌผ์ ์์ฑํฉ๋๋ค.

```shell
$ python demo_preprocessing.py --data_dir "./camera_dataset/" --num_samples 10 --save_dir './sample_preprocessing/'
```



### Outputs

```shell
[SAVE_DIR]
โโโ original/ # ์๋ณธ ์ด๋ฏธ์ง
โโโ hinet/ # HINet์ ์ํ data preprocessing ๊ฒฐ๊ณผ๋ฌผ
โโโ pix2pix/ # pix2pix๋ฅผ ์ํ data preprocessing ๊ฒฐ๊ณผ๋ฌผ
```

### Arguments

`data_dir`: input ๋ฐ์ดํฐ ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก

`num_samples`: ์์ฑํ  ์ํ ์

`save_dir`: Augmentation ์ ์ฉ ๊ฒฐ๊ณผ๋ฅผ ์ ์ฅํ  ๋๋ ํ ๋ฆฌ ๊ฒฝ๋ก

`stride`: Sliding Window ์ ์ฌ์ฉํ  stride

`patch_size`: Sliding Window ์ ์ฌ์ฉํ  patch ์ฌ์ด์ฆ
