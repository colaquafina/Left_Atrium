import os
from glob import glob
import random
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    SpatialPadd,
    EnsureTyped,
)

from config import (
    LABEL_ROOT,
    UNLABEL_ROOT,
    PIXDIM,
    A_MIN,
    A_MAX,
    B_MIN,
    B_MAX,
    BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    VAL_NUM_WORKERS,
    CACHE_RATE,
    RANDOM_STATE,
    IMG_SIZE,
)


# -------------------------------------------------------------------
# Set your unlabeled data root here
# -------------------------------------------------------------------
# UNLABELED_ROOT = "/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/unlabel_data"


# -------------------------------------------------------------------
# Labeled data
# -------------------------------------------------------------------
def split_data_dicts(data_dicts, test_size=0.2, random_state=RANDOM_STATE):
    shuffled = list(data_dicts)
    rng = random.Random(random_state)
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * test_size)))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def get_data_dicts(root=LABEL_ROOT):
    case_dirs = sorted(glob(os.path.join(root, "train_*")))
    data_dicts = []

    for case_dir in case_dirs:
        image_files = sorted(glob(os.path.join(case_dir, "[0-9]*.nii.gz")))
        label_files = sorted(glob(os.path.join(case_dir, "label_*.nii.gz")))

        if len(image_files) != 1:
            raise ValueError(f"Expected 1 image in {case_dir}, found {len(image_files)}")
        if len(label_files) != 1:
            raise ValueError(f"Expected 1 label in {case_dir}, found {len(label_files)}")

        data_dicts.append({
            "image": image_files[0],
            "label": label_files[0],
        })

    return data_dicts


# -------------------------------------------------------------------
# Unlabeled data
# -------------------------------------------------------------------
def get_unlabeled_data_dicts(root=UNLABEL_ROOT):
    """
    Assumes each case folder contains exactly one image .nii.gz and no label file.
    Example:
        unlabel_data/train_051/051.nii.gz
        unlabel_data/train_052/052.nii.gz
    """
    case_dirs = sorted(glob(os.path.join(root, "*")))
    data_dicts = []

    for case_dir in case_dirs:
        if not os.path.isdir(case_dir):
            continue

        image_files = [
            f for f in sorted(glob(os.path.join(case_dir, "*.nii.gz")))
            if "label" not in os.path.basename(f).lower()
        ]

        if len(image_files) != 1:
            raise ValueError(f"Expected 1 unlabeled image in {case_dir}, found {len(image_files)}")

        data_dicts.append({
            "image": image_files[0],
        })

    return data_dicts


# -------------------------------------------------------------------
# Transforms for labeled training
# -------------------------------------------------------------------
def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=IMG_SIZE),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMG_SIZE,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])


# -------------------------------------------------------------------
# Transforms for unlabeled training
# -------------------------------------------------------------------
def get_unlabeled_train_transforms():
    """
    For unlabeled data, we cannot use RandCropByPosNegLabeld because there is no label.
    So we use random spatial crops instead.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=PIXDIM,
            mode=("bilinear",),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=IMG_SIZE),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=IMG_SIZE,
            num_samples=4,
            random_size=False,
            random_center=True,
        ),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image"]),
    ])


# -------------------------------------------------------------------
# Validation transforms
# -------------------------------------------------------------------
def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


# -------------------------------------------------------------------
# Test transforms
# -------------------------------------------------------------------
def get_test_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=PIXDIM,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=B_MIN,
            b_max=B_MAX,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


# # -------------------------------------------------------------------
# # Original supervised loaders
# # -------------------------------------------------------------------
# def get_loaders():
#     data_dicts = get_data_dicts(LABEL_ROOT)
#     train_files, val_files = train_test_split(
#         data_dicts,
#         test_size=0.2,
#         random_state=RANDOM_STATE,
#     )

#     train_ds = CacheDataset(
#         data=train_files,
#         transform=get_train_transforms(),
#         cache_rate=CACHE_RATE,
#         num_workers=TRAIN_NUM_WORKERS,
#     )
#     val_ds = CacheDataset(
#         data=val_files,
#         transform=get_val_transforms(),
#         cache_rate=CACHE_RATE,
#         num_workers=VAL_NUM_WORKERS,
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=TRAIN_NUM_WORKERS,
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=VAL_NUM_WORKERS,
#         pin_memory=True,
#     )
#     return train_loader, val_loader


# -------------------------------------------------------------------
# Semi-supervised loaders for CPS
# -------------------------------------------------------------------
def get_ssl_loaders(labeled_root=LABEL_ROOT, unlabeled_root=UNLABEL_ROOT):
    labeled_data_dicts = get_data_dicts(labeled_root)
    unlabeled_data_dicts = get_unlabeled_data_dicts(unlabeled_root)

    train_files, val_files = split_data_dicts(
        labeled_data_dicts,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    labeled_train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=CACHE_RATE,
        num_workers=TRAIN_NUM_WORKERS,
    )

    unlabeled_train_ds = CacheDataset(
        data=unlabeled_data_dicts,
        transform=get_unlabeled_train_transforms(),
        cache_rate=CACHE_RATE,
        num_workers=TRAIN_NUM_WORKERS,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=CACHE_RATE,
        num_workers=VAL_NUM_WORKERS,
    )

    labeled_loader = DataLoader(
        labeled_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    unlabeled_loader = DataLoader(
        unlabeled_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=True,
    )

    return labeled_loader, unlabeled_loader, val_loader


# -------------------------------------------------------------------
# Test loader
# -------------------------------------------------------------------
def get_test_loader(root=LABEL_ROOT):
    data_dicts = get_data_dicts(root)

    test_ds = CacheDataset(
        data=data_dicts,
        transform=get_test_transforms(),
        cache_rate=CACHE_RATE,
        num_workers=VAL_NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader
