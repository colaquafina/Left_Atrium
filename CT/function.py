import logging
import os
import matplotlib.pyplot as plt


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.txt")

    logger = logging.getLogger("CPS_Training")
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers if rerun
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_segmentation_preview(
    image,
    label,
    pred,
    epoch,
    save_dir,
    prefix="train",
    slice_indices=None,
    num_slices=5,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    image_np = image[0, 0].detach().cpu().numpy()
    label_np = label[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    depth = image_np.shape[-1]

    # 👉 choose slices
    if slice_indices is None:
        slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    n_slices = len(slice_indices)

    fig, axes = plt.subplots(n_slices, 3, figsize=(9, 3 * n_slices))

    for i, idx in enumerate(slice_indices):
        img_slice = image_np[:, :, idx]
        label_slice = label_np[:, :, idx]
        pred_slice = pred_np[:, :, idx]

        axes[i, 0].imshow(img_slice, cmap="gray")
        axes[i, 0].set_title(f"Image (z={idx})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_slice, cmap="gray")
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_slice, cmap="gray")
        axes[i, 2].set_title("Pred")
        axes[i, 2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"{prefix}_epoch_{epoch:03d}_multislice.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()