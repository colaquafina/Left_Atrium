import numpy as np
from torch import nn
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
# import kornia
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def save_training_preview(epoch, image, gt_la, gt_scar, pred_la, pred_scar, tag, preview_dir):
    image_np = image[0, 0].detach().cpu().numpy()
    gt_la_np = gt_la[0, 0].detach().cpu().numpy()
    gt_scar_np = (gt_scar[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    pred_la_np = (pred_la[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    pred_scar_prob_np = pred_scar[0, 1].detach().cpu().numpy()
    # pred_scar_np = ((pred_scar_prob_np > 0.5) & (pred_la_np > 0)).astype(np.uint8)
    # Keep top 50% scar probabilities inside predicted LA
    scar_vals_in_la = pred_scar_prob_np[pred_la_np > 0]

    pred_scar_np = (pred_scar_prob_np > 0.5).astype(np.uint8)
    
    scar_volume = gt_scar_np.sum(axis=(0, 1))
    if scar_volume.max() > 0:
        slice_idx = int(np.argmax(scar_volume))
    else:
        slice_idx = image_np.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panel_data = [
        (image_np[:, :, slice_idx], 'Image', 'gray'),
        (gt_la_np[:, :, slice_idx], 'GT LA', 'viridis'),
        (pred_la_np[:, :, slice_idx], 'Pred LA', 'viridis'),
        (gt_scar_np[:, :, slice_idx], 'GT Scar', 'magma'),
        (pred_scar_np[:, :, slice_idx], 'Pred Scar', 'magma'),
        (pred_scar_prob_np[:, :, slice_idx], 'Pred Scar Prob', 'magma'),
    ]

    for ax, (arr, title, cmap) in zip(axes.flat, panel_data):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    fig.suptitle(f'{tag} epoch {epoch} slice {slice_idx}', fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(preview_dir, f'{tag}_epoch_{epoch:04d}_slice_{slice_idx:02d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    
def _load_state_dict_safely(net_param):
    try:
        return torch.load(net_param, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(net_param, map_location='cpu')


def boundary_band(mask, kernel_size=5):
    mask = (mask > 0.5).float()

    dilated = F.max_pool3d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    eroded = -F.max_pool3d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return (dilated - eroded).clamp(0.0, 1.0)

def soft_boundary(probability, kernel_size=5):
    local_max = F.max_pool3d(
        probability,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    local_min = -F.max_pool3d(
        -probability,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return (local_max - local_min).clamp(min=0.0, max=1.0)

def shape_attention_losses(out_la,out_scar,label,prob_normal,prob_scar):
    gt_difference = prob_normal - prob_scar
    pred_difference = (
        out_scar[:, 0:1] - out_scar[:, 1:2]
    )

    # M1: ground-truth LA boundary
    mask_gd = boundary_band(label)

    # M2: differentiable predicted LA boundary
    mask_pred = soft_boundary(out_la)

    squared_error = (
        pred_difference - gt_difference
    ).square()

    loss_scar_mask1 = torch.sum(
        mask_gd * squared_error
    ) / torch.clamp(
        torch.sum(mask_gd),
        min=1.0,
    )

    loss_scar_mask2 = torch.sum(
        mask_pred * squared_error
    ) / torch.clamp(
        torch.sum(mask_pred),
        min=1.0,
    )

    return loss_scar_mask1, loss_scar_mask2


def F_loss_scar(output, label, LAdist, prob_normal, prob_scar):
    out_LA, out_scar = output
    lossfunc1 = nn.BCELoss().to(device)
    loss_la = lossfunc1(out_LA, label)
    loss_sdf_la = torch.mean(((out_LA-0.5)*LAdist))

    lossfunc2 = nn.MSELoss().to(device)
    gt_scar_probmap = torch.cat((prob_normal, prob_scar), dim=1)
    loss_scar = lossfunc2(out_scar, gt_scar_probmap)#F_hellinger_distance
    
    loss_scar_mask1, loss_scar_mask2 = shape_attention_losses(out_la=out_LA, out_scar=out_scar, label=label, prob_normal=prob_normal, prob_scar=prob_scar,)
        
    return loss_la, loss_sdf_la, loss_scar, loss_scar_mask1, loss_scar_mask2


def F_mkdir(path):
    os.makedirs(path, exist_ok=True)

def F_hellinger_distance(p, q):
    """
    Calculates the hellinger's distance between two probability distributions.
    p --> probability vector 1.
    q --> probability vector 2. 
    """
    #d = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) **2)) / np.sqrt(2)
    # d = torch.mean(((torch.sqrt(p) - torch.sqrt(q)) **2) / np.sqrt(2))
    lossfunc2 = nn.MSELoss().to(device)
    d = lossfunc2(torch.sqrt(p), torch.sqrt(q))/ np.sqrt(2)

    return d


def F_DistTransform(lab):
    posmask = lab.astype(bool)
    if posmask.any():
        negmask = ~posmask
        fg_dtm = distance(negmask)
    else:
        # No foreground voxels for this class in the crop.
        # Returning +inf makes exp(-fg_dtm) become 0 everywhere downstream.
        fg_dtm = np.full(lab.shape, np.inf, dtype=np.float32)
    return fg_dtm

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    T = 50
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = T*np.ones(out_shape) #np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                #sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return np.clip(normalized_sdf, -T, T)



def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, 1, n_1, ..., n_k)
    :param B: (n_batch, 1, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    return F_Dice(torch.cat([1 - torch.clamp(torch.abs(A - i), 0, 1) for i in class_labels], 1),
                torch.cat([1 - torch.clamp(torch.abs(B - i), 0, 1) for i in class_labels], 1))


def F_DistTransformMap(img_gt):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """
    posmask = img_gt.astype(bool)
    if posmask.any():
        fg_dtm = distance(posmask)
    else:
        fg_dtm = np.zeros(img_gt.shape, dtype=np.float32)
    return fg_dtm

def F_Dice(A, B):
    '''
    A: (n_batch, n_class, ...)
    B: (n_batch, n_class, ...)
    return: (n_batch, n_class)
    '''
    eps = 1e-8
#    assert torch.sum(A * (1 - A)).abs().item() < eps and torch.sum(B * (1 - B)).abs().item() < eps
    A = A.flatten(2).float(); B = B.flatten(2).float()
    ABsum = A.sum(-1) + B.sum(-1)
    return 2 * torch.sum(A * B, -1) / (ABsum + eps)




def binary_dice_score(pred, target, threshold=0.5):
    '''
    pred: (n_batch, n_class, ...)
    target: (n_batch, n_class, ...)
    return: (n_batch, n_class)
    '''
    eps = 1e-8

    # flatten first
    pred = (pred > threshold).float().flatten(2)  # (B, C, N)
    target = (target > threshold).float().flatten(2)

    intersection = torch.sum(pred * target, dim=-1)
    denominator = torch.sum(pred, dim=-1) + torch.sum(target, dim=-1)

    return (2 * intersection + eps) / (denominator + eps)


#-----------------load net param-----------------------------
def F_LoadsubParam(net_param, sub_net, target_net):
    print(net_param)
    state_dict = _load_state_dict_safely(net_param)
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    sub_net.load_state_dict(new_state_dict)

    # ---------------load the param of Seg_net into SSM_net---------------
    sourceDict = sub_net.state_dict()
    targetDict = target_net.state_dict()
    target_net.load_state_dict({k: sourceDict[k] if k in sourceDict else targetDict[k] for k in targetDict})

def F_LoadParam(net_param, target_net):
    print(net_param)
    state_dict = _load_state_dict_safely(net_param)
    target_net.load_state_dict(state_dict)

def F_LoadParam_test(net_param, target_net):
    print(net_param)
    state_dict = _load_state_dict_safely(net_param)

    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    target_net.load_state_dict(new_state_dict)
