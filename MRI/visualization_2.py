import re
import numpy as np
import pylab as pl

log_path = "/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/logs/Gson_AJSQ_47997835.out"

number = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf"

train_pattern = re.compile(
    rf"Train epoch\s+(\d+),\s+avg loss:\s+({number}),\s+avg LA Dice:\s+({number}),\s+avg scar Dice:\s+({number})",
    re.IGNORECASE
)

val_pattern = re.compile(
    rf"Validation epoch\s+(\d+),\s+loss:\s+({number}),\s+LA loss:\s+({number}),\s+scar loss:\s+({number}),\s+"
    rf"avg LA Dice:\s+({number}),\s+avg scar Dice:\s+({number})",
    re.IGNORECASE
)

train_epoch, train_loss, train_la_dice, train_scar_dice = [], [], [], []
val_epoch, val_loss, val_la_dice, val_scar_dice = [], [], [], []

unmatched_train_lines = []
unmatched_val_lines = []

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("Train epoch"):
            m = train_pattern.search(line)
            if m:
                train_epoch.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_la_dice.append(float(m.group(3)))
                train_scar_dice.append(float(m.group(4)))
            else:
                unmatched_train_lines.append(line)

        elif line.startswith("Validation epoch"):
            m = val_pattern.search(line)
            if m:
                val_epoch.append(int(m.group(1)))
                val_loss.append(float(m.group(2)))
                val_la_dice.append(float(m.group(5)))
                val_scar_dice.append(float(m.group(6)))
            else:
                unmatched_val_lines.append(line)

print(f"Found {len(train_epoch)} training epochs")
print(f"Found {len(val_epoch)} validation epochs")

print(f"Unmatched train lines: {len(unmatched_train_lines)}")
print(f"Unmatched validation lines: {len(unmatched_val_lines)}")

if len(unmatched_train_lines) > 0:
    print("\nExample unmatched train lines:")
    for x in unmatched_train_lines[:5]:
        print(x)

if len(unmatched_val_lines) > 0:
    print("\nExample unmatched validation lines:")
    for x in unmatched_val_lines[:5]:
        print(x)

train_epoch = np.array(train_epoch)
train_loss = np.array(train_loss)
train_la_dice = np.array(train_la_dice)
train_scar_dice = np.array(train_scar_dice)

val_epoch = np.array(val_epoch)
val_loss = np.array(val_loss)
val_la_dice = np.array(val_la_dice)
val_scar_dice = np.array(val_scar_dice)

pl.figure(figsize=(12, 4))

pl.subplot(1, 3, 1)
pl.plot(train_epoch, train_loss, "b-", label="Train Loss")
pl.plot(val_epoch, val_loss, "r--", label="Validation Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")
pl.title("Train vs Validation Loss")
pl.legend(frameon=False)

pl.subplot(1, 3, 2)
pl.plot(train_epoch, train_la_dice, "b-", label="Train LA Dice")
pl.plot(val_epoch, val_la_dice, "r--", label="Validation LA Dice")
pl.xlabel("Epoch")
pl.ylabel("Dice")
pl.title("Train vs Validation LA Dice")
pl.ylim(0, 1)
pl.legend(frameon=False)

pl.subplot(1, 3, 3)
pl.plot(train_epoch, train_scar_dice, "b-", label="Train Scar Dice")
pl.plot(val_epoch, val_scar_dice, "r--", label="Validation Scar Dice")
pl.xlabel("Epoch")
pl.ylabel("Dice")
pl.title("Train vs Validation Scar Dice")
pl.ylim(0, 1)
pl.legend(frameon=False)

pl.tight_layout()
pl.savefig("train_validation_from_log.png", dpi=300)
pl.show()