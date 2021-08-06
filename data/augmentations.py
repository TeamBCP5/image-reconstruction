import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transform(transform_type: str = "pix2pix") -> A.Compose:
    if transform_type == "pix2pix":
        transform = A.Compose(
            [
                A.Cutout(
                    num_holes=8,
                    max_h_size=6,
                    max_w_size=6,
                    fill_value=[255, 0, 0],
                    always_apply=False,
                    p=0.6,
                ),
                A.Cutout(
                    num_holes=8,
                    max_h_size=6,
                    max_w_size=6,
                    fill_value=[0, 255, 0],
                    always_apply=False,
                    p=0.6,
                ),
                A.Cutout(
                    num_holes=8,
                    max_h_size=6,
                    max_w_size=6,
                    fill_value=[0, 0, 255],
                    always_apply=False,
                    p=0.6,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    max_pixel_value=255.0,
                    always_apply=False,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            additional_targets={"image": "image", "label": "image"},
            p=1.0,
        )

    elif transform_type == "hinet":
        transform = A.Compose(
            [
                A.Resize(1224, 1632),
                A.HorizontalFlip(p=0.5),
                A.ChannelShuffle(p=0.5),
                ToTensorV2(p=1.0),
            ],
            additional_targets={"label": "image"},
            p=1.0,
        )
    else:
        raise NotImplementedError(f"There's no transform type named '{transform_type}'")

    return transform


def get_valid_transform(transform_type: str = "pix2pix") -> A.Compose:
    if transform_type == "pix2pix":
        transform = A.Compose(
            [
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    max_pixel_value=255.0,
                    always_apply=False,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            additional_targets={"source": "image", "label": "image"},
            p=1.0,
        )

    elif transform_type == "hinet":
        transform = A.Compose(
            [
                A.Resize(1224, 1632),
                ToTensorV2(p=1.0),
            ],
            additional_targets={"label": "image"},
        )

    elif transform_type == "inference":
        transform = A.Compose(
            [
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(transpose_mask=True, p=1.0),
            ],
            additional_targets={"image": "image", "label": "image"},
            p=1.0,
        )
    else:
        raise NotImplementedError(f"There's no transform type named '{transform_type}'")

    return transform
