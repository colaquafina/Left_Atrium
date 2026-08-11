import numpy as np
import nibabel as nib
import torch
from scipy import stats
from function import compute_sdf
# import kornia
from function import F_DistTransform
# from scipy.special import expit, logit
# from skimage.exposure import match_histograms

np.seterr(divide='ignore', invalid='ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height = depth = 256 # the original size is 576 576
length = 44 #the total slice is 44
patch_size = (height, depth, length)

from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandBiasFieldd,
)

def get_train_augmentation():
    return Compose([
        RandAffined(
            keys=["image", "label", "scar"],
            prob=0.1,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.03, 0.03, 0.03),
            translate_range=(2, 2, 1),
            mode=("bilinear", "nearest", "nearest"),
            padding_mode="border",
        ),

        RandAdjustContrastd(
            keys=["image"],
            gamma=(0.8, 1.2),
            prob=0.2,
        ),

        RandBiasFieldd(
            keys=["image"],
            coeff_range=(0.0, 0.2),
            prob=0.2,
        ),

        RandGaussianNoised(
            keys=["image"],
            prob=0.15,
            mean=0.0,
            std=0.03,
        ),

        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.25, 0.75),
        ),
    ])

def apply_augmentation(image, label, scar, aug=None):
    if aug is None:
        return image, label, scar

    data = {
        "image": image[None, ...].astype(np.float32),
        "label": label[None, ...].astype(np.float32),
        "scar": scar[None, ...].astype(np.float32),
    }

    original_scar_sum = data["scar"].sum()

    data_aug = aug(data)

    image_aug = np.asarray(data_aug["image"][0])
    label_aug = np.asarray(data_aug["label"][0])
    scar_aug = np.asarray(data_aug["scar"][0])

    label_aug = (label_aug > 0.5).astype(np.uint8)
    scar_aug = (scar_aug > 0.5).astype(np.uint8)

    # Do not remove scar outside the LA.
    augmented_scar_sum = scar_aug.sum()

    # Reject a spatial transformation if it destroys too much scar.
    if original_scar_sum > 0:
        remaining_fraction = augmented_scar_sum / original_scar_sum

        if remaining_fraction < 0.8:
            return (
                image.astype(np.float32),
                label.astype(np.uint8),
                scar.astype(np.uint8),
            )

    return (
        image_aug.astype(np.float32),
        label_aug,
        scar_aug,
    )

def _get_la_center_coord(numpylabel):
    center_slice = numpylabel[:, :, int(numpylabel.shape[2] / 2)]
    coords = np.where(center_slice > 0)
    if len(coords[0]) == 0:
        raise ValueError("LA label is empty on the center slice. Cannot compute MRI-style center crop.")
    return np.floor(np.mean(np.stack(coords), axis=-1)).astype(np.int16)


def _get_image_center_coord(numpyimage):
    return np.array(
        [numpyimage.shape[0] // 2, numpyimage.shape[1] // 2],
        dtype=np.int16
    )


def F_nifity_imageCrop(numpyimage, center_coord, output_size=patch_size):
    """
    MRI-style crop:
    - determine crop center from the middle LA slice
    - crop in x/y only
    - keep the full z dimension
    """
    crop_h, crop_w, _ = output_size
    center_x, center_y = center_coord
    full_depth = numpyimage.shape[2]

    sx = int(center_x - crop_h / 2)
    ex = sx + crop_h
    sy = int(center_y - crop_w / 2)
    ey = sy + crop_w

    crop = np.zeros((crop_h, crop_w, full_depth), dtype=np.float32)

    src_x0 = max(sx, 0)
    src_x1 = min(ex, numpyimage.shape[0])
    src_y0 = max(sy, 0)
    src_y1 = min(ey, numpyimage.shape[1])

    dst_x0 = src_x0 - sx
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = src_y0 - sy
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    crop[dst_x0:dst_x1, dst_y0:dst_y1, :] = numpyimage[src_x0:src_x1, src_y0:src_y1, :]

    return crop

# def F_nifity_bboxCrop(numpyimage, numpylabel, margin=0, output_size=patch_size):
#     """
#     Crop around the full 3D LA bounding box with margin.
#     This is safer than center-slice cropping.

#     Args:
#         numpyimage: MRI image or label, shape (H, W, D)
#         numpylabel: LA label used to determine bounding box, shape (H, W, D)
#         margin: extra voxels around LA
#         output_size: final fixed crop size, e.g. (192, 192, 80)

#     Returns:
#         cropped image/label with shape output_size
#     """

#     coords = np.where(numpylabel > 0)

#     if len(coords[0]) == 0:
#         raise ValueError("LA label is empty. Cannot compute bounding box crop.")

#     x_min, x_max = coords[0].min(), coords[0].max()
#     y_min, y_max = coords[1].min(), coords[1].max()
#     z_min, z_max = coords[2].min(), coords[2].max()

#     # add margin
#     x_min -= margin
#     x_max += margin
#     y_min -= margin
#     y_max += margin
#     # z_min -= margin
#     # z_max += margin
    
#     crop_h, crop_w, crop_d = output_size
    
#     bbox_size = (
#         x_max - x_min + 1,
#         y_max - y_min + 1,
#         z_max - z_min + 1
#     )

#     # if bbox_size[0] > crop_h or bbox_size[1] > crop_w or bbox_size[2] > crop_d:
#     #     print("WARNING: LA bounding box is larger than crop size.")
#     #     print(bbox_size)
#     #     print(patch_size)

#     # bounding box center
#     cx = int(round((x_min + x_max) / 2))
#     cy = int(round((y_min + y_max) / 2))
#     cz = int(round((z_min + z_max) / 2))

#     # fixed-size crop range
#     sx = cx - crop_h // 2
#     ex = sx + crop_h

#     sy = cy - crop_w // 2
#     ey = sy + crop_w

#     sz = cz - crop_d // 2
#     ez = sz + crop_d

#     # pad image if crop goes outside boundary
#     pad_x_before = max(0, -sx)
#     pad_y_before = max(0, -sy)
#     pad_z_before = max(0, -sz)

#     pad_x_after = max(0, ex - numpyimage.shape[0])
#     pad_y_after = max(0, ey - numpyimage.shape[1])
#     pad_z_after = max(0, ez - numpyimage.shape[2])

#     numpyimage_pad = np.pad(
#         numpyimage,
#         ((pad_x_before, pad_x_after),
#          (pad_y_before, pad_y_after),
#          (pad_z_before, pad_z_after)),
#         mode="constant",
#         constant_values=0
#     )

#     sx += pad_x_before
#     ex += pad_x_before
#     sy += pad_y_before
#     ey += pad_y_before
#     sz += pad_z_before
#     ez += pad_z_before

#     crop = numpyimage_pad[sx:ex, sy:ey, sz:ez]

#     return crop.astype(np.float32)


def _build_scar_targets(numpylabel_crop, numpyscarlabel_crop):
    la_mask = numpylabel_crop > 0
    scar_mask = numpyscarlabel_crop > 0
    normal_mask = la_mask & (~scar_mask)

    gt_dis_normal = F_DistTransform(normal_mask)
    gt_dis_scar = F_DistTransform(scar_mask)
    gt_dis_normal = np.expand_dims(np.exp(-gt_dis_normal), 0)
    gt_dis_scar = np.expand_dims(np.exp(-gt_dis_scar), 0)

    return gt_dis_normal, gt_dis_scar

def LoadDataset_scar(imagenames, labelnames, scarlabelnames, augment=False):

    niblabel = nib.load(labelnames)
    labeldata = np.asanyarray(niblabel.dataobj)
    numpylabel = np.array(labeldata).squeeze()

    nibimage = nib.load(imagenames)
    imagedata = np.asanyarray(nibimage.dataobj)
    numpyimage = np.array(imagedata).squeeze()

    nibscarlabel = nib.load(scarlabelnames)
    scarlabeldata = np.asanyarray(nibscarlabel.dataobj)
    numpyscarlabel = np.array(scarlabeldata).squeeze()

    # 1. MRI-style center crop first
    center_coord = _get_la_center_coord(numpylabel)
    numpylabel_crop = F_nifity_imageCrop(numpylabel, center_coord)
    numpyimage_crop = F_nifity_imageCrop(numpyimage, center_coord)
    numpyscarlabel_crop = F_nifity_imageCrop(numpyscarlabel, center_coord)

    # 2. augmentation
    if augment:
        aug = get_train_augmentation()
        numpyimage_crop, numpylabel_crop, numpyscarlabel_crop = apply_augmentation(
            numpyimage_crop,
            numpylabel_crop,
            numpyscarlabel_crop,
            aug=aug
        )

    # 3. normalize image after augmentation
    numpyimage_crop_processed = np.nan_to_num(stats.zscore(numpyimage_crop))

    # 4. build LA SDF
    numpylabel_crop_new = np.expand_dims(numpylabel_crop, 0)
    numpylabel_crop_new = np.expand_dims(numpylabel_crop_new, 0)
    numpylabel_crop_new = (numpylabel_crop_new > 0) * 1

    gt_dis = compute_sdf(numpylabel_crop_new, numpylabel_crop_new.shape)
    gt_LA_dis = np.squeeze(gt_dis, axis=1)

    # 5. build scar soft targets after augmentation
    gt_dis_normal, gt_dis_scar = _build_scar_targets(
        numpylabel_crop,
        numpyscarlabel_crop
    )

    return (
        np.expand_dims(numpyimage_crop_processed, 0),
        np.expand_dims(numpylabel_crop, 0),
        gt_LA_dis,
        gt_dis_normal,
        gt_dis_scar
    )


def LoadTestDataset(imagename):
    nibimage = nib.load(imagename)
    imagedata = np.asanyarray(nibimage.dataobj)
    numpyimage = np.array(imagedata).squeeze()

    center_coord = _get_image_center_coord(numpyimage)
    numpyimage_crop = F_nifity_imageCrop(numpyimage, center_coord)
    numpyimage_crop_processed = np.nan_to_num(stats.zscore(numpyimage_crop))

    return np.expand_dims(numpyimage_crop_processed, 0), nibimage, center_coord


def save_test_img(nibimage, outputlab, center_coord=None):
    outputlab = np.asarray(outputlab)
    imagedata = np.asanyarray(nibimage.dataobj)
    reference = np.array(imagedata).squeeze()
    ref_shape = reference.shape

    if center_coord is None:
        center_coord = _get_image_center_coord(reference)
    crop_h, crop_w = outputlab.shape[0], outputlab.shape[1]
    sx = int(center_coord[0] - crop_h / 2)
    sy = int(center_coord[1] - crop_w / 2)
    ex = sx + crop_h
    ey = sy + crop_w

    restored = np.zeros(ref_shape, dtype=outputlab.dtype)

    src_x0 = max(0, -sx)
    src_y0 = max(0, -sy)
    dst_x0 = max(sx, 0)
    dst_y0 = max(sy, 0)
    copy_x = min(ex, ref_shape[0]) - dst_x0
    copy_y = min(ey, ref_shape[1]) - dst_y0
    copy_z = min(outputlab.shape[2], ref_shape[2])

    if copy_x > 0 and copy_y > 0 and copy_z > 0:
        restored[
            dst_x0:dst_x0 + copy_x,
            dst_y0:dst_y0 + copy_y,
            :copy_z
        ] = outputlab[
            src_x0:src_x0 + copy_x,
            src_y0:src_y0 + copy_y,
            :copy_z
        ]

    predictlabel = nib.Nifti1Image(restored, nibimage.affine, nibimage.header)
    return predictlabel

def ProcessTestDataset(imagename, Seg_net):
    # print('loading test image: ' + imagename)

    numpyimage, nibimage, center_coord = LoadTestDataset(imagename)

    tensorimage = torch.from_numpy(numpyimage).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out_LA, out_scar = Seg_net(tensorimage)

    out_scar = out_scar * ((out_scar>0.1).float())
    # LA prediction
    output1 = np.squeeze(out_LA.cpu().detach().numpy(), axis=(0, 1))
    label_LA = (output1 > 0.5).astype(np.uint8)

    # Scar prediction
    output2 = np.squeeze(out_scar.cpu().detach().numpy(), axis=0)
    output_new = np.argmax(output2, axis=0)
    scar_region = (output_new == 1) & (output2[1] > 0.5)
    label_scar = scar_region.astype(np.uint8)

    predict_LA = save_test_img(nibimage, label_LA, center_coord=center_coord)
    predict_scar = save_test_img(nibimage, label_scar, center_coord=center_coord)

    return predict_LA, predict_scar
