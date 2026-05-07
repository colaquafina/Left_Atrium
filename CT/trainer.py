import os
import math
from monai.metrics import DiceMetric
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted
from config import IMG_SIZE, NUM_CLASSES
from function import save_segmentation_preview


post_pred = AsDiscreted(keys="pred", argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscreted(keys="label", to_onehot=NUM_CLASSES)


def weak_augment(x):
    # Keep weak augmentation geometry-preserving so pseudo labels stay aligned.
    return x


def strong_augment(x):
    if torch.rand(1).item() < 0.5:
        noise = 0.05 * torch.randn_like(x)
        x = x + noise

    if torch.rand(1).item() < 0.5:
        scale = 0.9 + 0.2 * torch.rand(
            x.shape[0], *([1] * (x.ndim - 1)), device=x.device
        )
        shift = 0.1 * (
            torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) - 0.5
        )
        x = x * scale + shift

    return x


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


def masked_cross_entropy_loss(logits, pseudo_label, mask):
    """
    logits: [B, C, ...]
    pseudo_label: [B, ...] long
    mask: [B, ...] bool/float
    """
    per_pixel_loss = F.cross_entropy(logits, pseudo_label, reduction="none")
    mask = mask.float()

    valid_pixels = mask.sum()
    if valid_pixels.item() < 1:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    return (per_pixel_loss * mask).sum() / (valid_pixels + 1e-8)



def train_one_epoch_cps(
    model1,
    model2,
    labeled_loader,
    unlabeled_loader,
    optimizer1,
    optimizer2,
    loss_function,
    device,
    epoch,
    max_cps_weight=1.0,
    rampup_epochs=30,
    confidence_threshold=0.95,
    preview_dir=None,
    save_preview_interval=5,
):
    model1.train()
    model2.train()

    epoch_total_loss = 0.0
    epoch_sup_loss = 0.0
    epoch_cps_loss = 0.0

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric.reset()

    cps_weight = max_cps_weight * sigmoid_rampup(epoch, rampup_epochs)

    unlabeled_iter = iter(unlabeled_loader)

    # only save one preview image per epoch
    preview_saved = False

    for batch_data_l in labeled_loader:
        try:
            batch_data_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            batch_data_u = next(unlabeled_iter)

        # -------------------------
        # labeled data
        # -------------------------
        inputs_l = batch_data_l["image"].to(device)
        labels_l = batch_data_l["label"].to(device)

        # -------------------------
        # unlabeled data
        # -------------------------
        inputs_u = batch_data_u["image"].to(device)

        # -------------------------
        # supervised branch
        # -------------------------
        outputs1_l = model1(inputs_l)
        outputs2_l = model2(inputs_l)

        loss_sup1 = loss_function(outputs1_l, labels_l)
        loss_sup2 = loss_function(outputs2_l, labels_l)
        loss_sup = loss_sup1 + loss_sup2

        # -------------------------
        # training dice + preview
        # -------------------------
        with torch.no_grad():
            prob1_l = torch.softmax(outputs1_l, dim=1)
            prob2_l = torch.softmax(outputs2_l, dim=1)

            prob_ensemble_l = (prob1_l + prob2_l) / 2.0
            pred_l = torch.argmax(prob_ensemble_l, dim=1, keepdim=True)

            dice_metric(y_pred=pred_l, y=labels_l)

            # if (
            #     preview_dir is not None
            #     and (epoch + 1) % save_preview_interval == 0
            #     and not preview_saved
            # ):
            #     save_segmentation_preview(
            #         image=inputs_l,
            #         label=labels_l,
            #         pred=pred_l,
            #         epoch=epoch + 1,
            #         save_dir=preview_dir,
            #         prefix="train",
            #     )
            #     preview_saved = True

        # -------------------------
        # CPS branch on unlabeled data
        # -------------------------
        inputs_u1 = weak_augment(inputs_u.clone())
        inputs_u2 = strong_augment(inputs_u.clone())

        outputs1_u = model1(inputs_u1)
        outputs2_u = model2(inputs_u2)

        with torch.no_grad():
            prob1 = torch.softmax(outputs1_u.detach(), dim=1)
            prob2 = torch.softmax(outputs2_u.detach(), dim=1)

            conf1, pseudo1 = torch.max(prob1, dim=1)
            conf2, pseudo2 = torch.max(prob2, dim=1)

            mask1 = conf1 > confidence_threshold
            mask2 = conf2 > confidence_threshold

        loss_cps1 = masked_cross_entropy_loss(outputs1_u, pseudo2, mask2)
        loss_cps2 = masked_cross_entropy_loss(outputs2_u, pseudo1, mask1)

        loss_cps = loss_cps1 + loss_cps2

        # -------------------------
        # total loss
        # -------------------------
        loss = loss_sup + cps_weight * loss_cps

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        epoch_total_loss += loss.item()
        epoch_sup_loss += loss_sup.item()
        epoch_cps_loss += loss_cps.item()

    n = len(labeled_loader)

    train_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return {
        "total_loss": epoch_total_loss / n,
        "sup_loss": epoch_sup_loss / n,
        "cps_loss": epoch_cps_loss / n,
        "train_dice": train_dice,
        "cps_weight": cps_weight,
    }

def validate_ensemble(
    model1,
    model2,
    loader,
    dice_metric,
    device,
    epoch=None,
    preview_dir=None,
):
    model1.eval()
    model2.eval()
    dice_metric.reset()

    preview_saved = False

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            outputs1 = sliding_window_inference(
                inputs,
                roi_size=IMG_SIZE,
                sw_batch_size=1,
                predictor=model1,
                overlap=0.5,
            )

            outputs2 = sliding_window_inference(
                inputs,
                roi_size=IMG_SIZE,
                sw_batch_size=1,
                predictor=model2,
                overlap=0.5,
            )

            outputs = 0.5 * (
                torch.softmax(outputs1, dim=1)
                + torch.softmax(outputs2, dim=1)
            )

            pred_dict = [{"pred": x} for x in decollate_batch(outputs)]
            pred_dict = [post_pred(x) for x in pred_dict]
            preds = [x["pred"] for x in pred_dict]

            label_dict = [{"label": x} for x in decollate_batch(labels)]
            label_dict = [post_label(x) for x in label_dict]
            labels_list = [x["label"] for x in label_dict]

            dice_metric(y_pred=preds, y=labels_list)

            # save only one validation preview per validation epoch
            if preview_dir is not None and not preview_saved:
                pred_label = torch.argmax(outputs, dim=1, keepdim=True)

                save_segmentation_preview(
                    image=inputs,
                    label=labels,
                    pred=pred_label,
                    epoch=epoch,
                    save_dir=preview_dir,
                    prefix="val",
                )

                preview_saved = True

    return dice_metric.aggregate().item()


def test_model_ensemble(model1, model2, loader, dice_metric, device):
    model1.eval()
    model2.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            outputs1 = sliding_window_inference(
                inputs,
                roi_size=IMG_SIZE,
                sw_batch_size=1,
                predictor=model1,
                overlap=0.5,
            )

            outputs2 = sliding_window_inference(
                inputs,
                roi_size=IMG_SIZE,
                sw_batch_size=1,
                predictor=model2,
                overlap=0.5,
            )

            outputs = 0.5 * (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1))

            pred_dict = [{"pred": x} for x in decollate_batch(outputs)]
            pred_dict = [post_pred(x) for x in pred_dict]
            preds = [x["pred"] for x in pred_dict]

            label_dict = [{"label": x} for x in decollate_batch(labels)]
            label_dict = [post_label(x) for x in label_dict]
            labels_list = [x["label"] for x in label_dict]

            dice_metric(y_pred=preds, y=labels_list)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


def save_checkpoint(model1, model2, optimizer1, optimizer2, save_path, epoch, best_metric):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model1": model1.state_dict(),
            "model2": model2.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        save_path,
    )
