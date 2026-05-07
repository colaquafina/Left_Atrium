import os
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import csv
from config import (
    OUTPUT_DIR, MAX_EPOCHS, VAL_INTERVAL, LR, WEIGHT_DECAY,
    MAX_CPS_WEIGHT, RAMPUP_EPOCHS, CONFIDENCE_THRESHOLD
)
from load_data import get_ssl_loaders, get_test_loader
from network import get_model
from trainer import (
    train_one_epoch_cps,
    validate_ensemble,
    save_checkpoint,
    test_model_ensemble,
)
from function import setup_logger

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(OUTPUT_DIR, "best_swin_unetr_ct_cps.pth"),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/train_data/label_data",
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger = setup_logger(OUTPUT_DIR)
    logger.info("Start program")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")
    
    csv_log_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")

    if args.mode == "train" and not os.path.exists(csv_log_path):
        with open(csv_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "total_loss",
                "sup_loss",
                "cps_loss",
                "train_dice",
            ])

    if args.mode == "train":
        labeled_loader, unlabeled_loader, val_loader = get_ssl_loaders()

        model1 = get_model().to(device)
        model2 = get_model().to(device)

        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        dice_metric = DiceMetric(include_background=False, reduction="mean")

        best_metric = -1
        best_metric_epoch = -1
        best_model_path = os.path.join(OUTPUT_DIR, "best_swin_unetr_ct_cps.pth")

        for epoch in range(MAX_EPOCHS):
            train_stats = train_one_epoch_cps(
                model1=model1,
                model2=model2,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                optimizer1=optimizer1,
                optimizer2=optimizer2,
                loss_function=loss_function,
                device=device,
                epoch=epoch,
                max_cps_weight=MAX_CPS_WEIGHT,
                rampup_epochs=RAMPUP_EPOCHS,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                preview_dir=os.path.join(OUTPUT_DIR, "training_previews"),
                save_preview_interval=VAL_INTERVAL,
            )

            logger.info(
                f"epoch {epoch+1}/{MAX_EPOCHS}, "
                f"total loss: {train_stats['total_loss']:.4f}, "
                f"sup loss: {train_stats['sup_loss']:.4f}, "
                f"cps loss: {train_stats['cps_loss']:.4f}, "
                f"train dice: {train_stats['train_dice']:.4f}, "
                f"cps weight: {train_stats['cps_weight']:.4f}"
            )
            
            with open(csv_log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_stats["total_loss"],
                    train_stats["sup_loss"],
                    train_stats["cps_loss"],
                    train_stats["train_dice"],
                ])

            if (epoch + 1) % VAL_INTERVAL == 0:
                metric = validate_ensemble(model1, model2, val_loader, dice_metric, device, epoch=epoch + 1, preview_dir=os.path.join(OUTPUT_DIR, "validation_previews"))
                logger.info(f"validation mean dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_checkpoint(
                        model1=model1,
                        model2=model2,
                        optimizer1=optimizer1,
                        optimizer2=optimizer2,
                        save_path=best_model_path,
                        epoch=epoch + 1,
                        best_metric=best_metric,
                    )
                    logger.info("saved new best metric model")

        logger.info(f"best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    elif args.mode == "test":
        model1 = get_model().to(device)
        model2 = get_model().to(device)

        checkpoint = torch.load(args.model_path, map_location=device)
        model1.load_state_dict(checkpoint["model1"])
        model2.load_state_dict(checkpoint["model2"])

        dice_metric = DiceMetric(include_background=False, reduction="mean")

        test_loader = get_test_loader(args.data_root)
        mean_dice = test_model_ensemble(model1, model2, test_loader, dice_metric, device)

        print("=" * 50)
        print(f"Test Mean Dice: {mean_dice:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
