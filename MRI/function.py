import numpy as np
from torch import nn
import torch
import re
import collections
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
# import kornia
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _find_latest_checkpoint(checkpoint_dir, prefix):
    pattern = os.path.join(checkpoint_dir, prefix + '*.pkl')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'No checkpoints found for pattern: {pattern}')

    def extract_epoch(path):
        match = re.search(r'(\d+)\.pkl$', os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=extract_epoch)


def save_training_preview(epoch, image, gt_la, gt_scar, pred_la, pred_scar, tag, preview_dir):
    image_np = image[0, 0].detach().cpu().numpy()
    gt_la_np = gt_la[0, 0].detach().cpu().numpy()
    gt_scar_np = (gt_scar[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    pred_la_np = (pred_la[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    pred_scar_prob_np = pred_scar[0, 1].detach().cpu().numpy()
    # pred_scar_np = ((pred_scar_prob_np > 0.5) & (pred_la_np > 0)).astype(np.uint8)
    # Keep top 50% scar probabilities inside predicted LA
    scar_vals_in_la = pred_scar_prob_np[pred_la_np > 0]

    if scar_vals_in_la.size > 0:
        if tag == 'train':
            thresh = 0.5
        else:
            thresh = np.max(scar_vals_in_la)*0.6
        pred_scar_np = ((pred_scar_prob_np > thresh) & (pred_la_np > 0)).astype(np.uint8)
        print(thresh)
    else:
        pred_scar_np = np.zeros_like(pred_scar_prob_np, dtype=np.uint8)
    
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
    
    
def get_consistency_weight(epoch):
    """
    Gradually turn on consistency loss.
    """
    ramp = min(1.0, epoch / 50)
    return 0.1 * ramp

def consistency_loss(student_output, teacher_output, SCAR_CONS_WEIGHT):
    """
    student_output: output from strong image
    teacher_output: output from weak image

    output = (LA_prediction, scar_prediction)
    """
    student_la, student_scar = student_output
    teacher_la, teacher_scar = teacher_output

    teacher_la = teacher_la.detach()
    teacher_scar = teacher_scar.detach()

    # LA consistency
    loss_cons_la = F.mse_loss(student_la, teacher_la)

    # Scar consistency only inside teacher-predicted LA region.
    # This avoids forcing scar predictions outside the atrium.
    la_mask = (teacher_la > 0.5).float()

    if la_mask.sum() > 0:
        loss_cons_scar = F.mse_loss(
            student_scar * la_mask,
            teacher_scar * la_mask
        )
    else:
        loss_cons_scar = torch.tensor(0.0, device=student_la.device)

    loss_cons = loss_cons_la + SCAR_CONS_WEIGHT * loss_cons_scar

    return loss_cons, loss_cons_la, loss_cons_scar


def _load_state_dict_safely(net_param):
    try:
        return torch.load(net_param, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(net_param, map_location='cpu')

def _boundary_mask_from_prob(prob_map, max_distance=1.5):
    """
    Build a thin boundary band from a soft target generated as exp(-distance).
    We exclude the region interior (prob ~= 1) and keep voxels within a small
    distance of the class boundary.
    """
    eps = 0.1
    lower_prob = np.exp(-max_distance)
    return ((prob_map >= lower_prob) & (prob_map < 1.0 - eps)).float()

def F_loss_scar(output, label, LAdist, prob_normal, prob_scar):
    out_LA, out_scar = output
    lossfunc1 = nn.BCELoss().to(device)
    loss_la = lossfunc1(out_LA, label)
    loss_sdf_la = torch.mean(((out_LA-0.5)*LAdist))

    lossfunc2 = nn.MSELoss().to(device)
    gt_scar_probmap = torch.cat((prob_normal, prob_scar), dim=1)
    loss_scar = lossfunc2(out_scar, gt_scar_probmap)#F_hellinger_distance

    lossfunc3 = nn.MSELoss(reduction='sum').to(device)
    normal_boundary = _boundary_mask_from_prob(prob_normal, max_distance=1.5)
    scar_boundary = _boundary_mask_from_prob(prob_scar, max_distance=1.5)
    mask_gd = ((normal_boundary + scar_boundary) > 0).float()
    # mask_gd = (torch.min(torch.abs(torch.logit(gt_scar_probmap)), dim=1)[0]==0).float()
    # mask_gd = (torch.min(-torch.log(gt_scar_probmap), dim=1)[0]==0).float()
    # out_LA_gradient = kornia.sobel(((out_LA>0.5).float()))
    # mask_pred = (out_LA_gradient>0.4).float()
    mask_pred = ((out_LA > 0.1) * (out_LA < 0.8)).float()
    mask_gd_denom = torch.clamp(torch.sum(mask_gd), min=1.0)
    mask_pred_denom = torch.clamp(torch.sum(mask_pred), min=1.0)
    loss_scar_mask1 = lossfunc3(mask_gd*(gt_scar_probmap[:, 0] - gt_scar_probmap[:, 1]), mask_gd*(out_scar[:, 0] - out_scar[:, 1])) / mask_gd_denom
    loss_scar_mask2 = lossfunc3(mask_pred * (gt_scar_probmap[:, 0] - gt_scar_probmap[:, 1]), mask_pred * (out_scar[:, 0] - out_scar[:, 1])) / mask_pred_denom

    return loss_la, loss_sdf_la, loss_scar, loss_scar_mask1, loss_scar_mask2

def F_mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)

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

def F_loss(output, label):

    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    weightedDice = 10*torch.mean(1-Dice[:, 1]) + 0.1*torch.mean(1-Dice[:, 0])
    Dice_loss = 1-weightedDice
    loss = CE_loss + 0.1*Dice_loss

    return loss

def F_loss_SDM(output, label):
    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    loss_seg = CE_loss

    gt_dis = compute_sdf(label.cpu().numpy(), output.shape)
    gt_dis = torch.from_numpy(gt_dis).float().to(device)
    loss_sdf_lei = torch.mean(((output - 0.5) * gt_dis))

    return loss_seg, loss_sdf_lei

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

def AAAI_sdf_loss(net_output, gt_sdm):
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([4, 1, 112, 112, 80])
    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    # L_SDF_AAAI = - L_product + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)
    L_SDF_AAAI = torch.norm(net_output - gt_sdm, 1) / torch.numel(net_output)

    return L_SDF_AAAI

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


def binary_dice_score_scar(pred, target, threshold=0.5):
    '''
    pred: (n_batch, n_class, ...)
    target: (n_batch, n_class, ...)
    return: (n_batch, n_class)
    '''
    eps = 1e-8

    # flatten
    pred_flat = pred.flatten(2)                     # (B, C, N)
    target = (target > 0.5).float().flatten(2)

    # max per (B, C)
    max_val = pred_flat.max(dim=-1, keepdim=True).values   # (B, C, 1)

    thresh = 0.6 * max_val

    # binarize
    pred_bin = (pred_flat > thresh).float()

    # Dice
    intersection = torch.sum(pred_bin * target, dim=-1)
    denominator = torch.sum(pred_bin, dim=-1) + torch.sum(target, dim=-1)

    return (2 * intersection + eps) / (denominator + eps)

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
