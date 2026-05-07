import pandas as pd
import re
import matplotlib.pyplot as plt

# =========================
# Load CSV (loss)
# =========================
csv_path = "/work/users/g/s/gsonw/BIOS740/final_project/CT/outputs/training_metrics.csv"
df = pd.read_csv(csv_path)

# =========================
# Parse log (dice)
# =========================
log_path = "/work/users/g/s/gsonw/BIOS740/final_project/CT/logs/swin_ct_47857659.err"

epochs_train = []
train_dices = []

epochs_val = []
val_dices = []

current_epoch = None

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        epoch_match = re.search(r"epoch\s+(\d+)/(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

            train_match = re.search(r"train dice:\s*([0-9.]+)", line)
            if train_match:
                train_dices.append(float(train_match.group(1)))
                epochs_train.append(current_epoch)

        val_match = re.search(r"validation mean dice:\s*([0-9.]+)", line)
        if val_match and current_epoch is not None:
            val_dices.append(float(val_match.group(1)))
            epochs_val.append(current_epoch)

# =========================
# Plot combined figure
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- Left: Loss ----
axes[0].plot(df["epoch"], df["total_loss"], label="Total Loss", color="tab:blue", linewidth=2)
axes[0].plot(df["epoch"], df["sup_loss"], label="Sup Loss", color="tab:green", linewidth=2)
axes[0].plot(df["epoch"], df["cps_loss"], label="CPS Loss", color="tab:orange", linewidth=2)

axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# ---- Right: Dice ----
axes[1].plot(
    epochs_train,
    train_dices,
    label="Train Dice",
    color="tab:blue",
    linewidth=2,
    alpha=0.8
)

axes[1].plot(
    epochs_val,
    val_dices,
    label="Validation Dice",
    color="tab:red",
    linewidth=2.5,
    linestyle="--",
    marker="o"
)

axes[1].set_title("Dice Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice Score")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

# ---- Final layout ----
plt.tight_layout()

save_path = "training_summary_combined.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved to:", save_path)