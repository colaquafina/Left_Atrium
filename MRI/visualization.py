import os
import re
import numpy as np
import pylab as pl

Root_DIR = '/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/lossfile'
log_path = '/work/users/g/s/gsonw/BIOS740/final_project/AJSQ_2/logs/Gson_AJSQ_47997835.out'

# =========================
# Read training txt files
# =========================
train_files = {
    'LA Loss': 'laLoss_3d.txt',
    'Scar Loss': 'scarLoss_3d.txt',
    'LA Dice': 'laDice_3d.txt',
    'Scar Dice': 'scarDice_3d.txt',
}

train_data = {}

for metric_name, filename in train_files.items():
    filepath = os.path.join(Root_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Missing training file: {filepath}")
        continue

    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    data = np.atleast_2d(data)

    train_data[metric_name] = {
        'epoch': data[:, 0],
        'value': data[:, 1],
    }


# =========================
# Read validation from log
# =========================
val_data = {
    'LA Loss': {'epoch': [], 'value': []},
    'Scar Loss': {'epoch': [], 'value': []},
    'LA Dice': {'epoch': [], 'value': []},
    'Scar Dice': {'epoch': [], 'value': []},
}

number = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf"

pattern = re.compile(
    rf"Validation epoch\s+(\d+),\s*"
    rf"loss:\s*({number}),\s*"
    rf"LA loss:\s*({number}),\s*"
    rf"scar loss:\s*({number}),\s*"
    rf"avg LA Dice:\s*({number}),\s*"
    rf"avg scar Dice:\s*({number})",
    re.IGNORECASE
)

unmatched_val_lines = []

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("Validation epoch"):
            match = pattern.search(line)

            if match:
                epoch = int(match.group(1))

                val_data['LA Loss']['epoch'].append(epoch)
                val_data['LA Loss']['value'].append(float(match.group(3)))

                val_data['Scar Loss']['epoch'].append(epoch)
                val_data['Scar Loss']['value'].append(float(match.group(4)))

                val_data['LA Dice']['epoch'].append(epoch)
                val_data['LA Dice']['value'].append(float(match.group(5)))

                val_data['Scar Dice']['epoch'].append(epoch)
                val_data['Scar Dice']['value'].append(float(match.group(6)))
            else:
                unmatched_val_lines.append(line)

for key in val_data:
    val_data[key]['epoch'] = np.array(val_data[key]['epoch'])
    val_data[key]['value'] = np.array(val_data[key]['value'])

print(f"Found validation epochs: {len(val_data['LA Dice']['epoch'])}")
print(f"Unmatched validation lines: {len(unmatched_val_lines)}")

if len(unmatched_val_lines) > 0:
    print("\nExample unmatched validation lines:")
    for x in unmatched_val_lines[:10]:
        print(x)


# =========================
# Plot train + validation
# =========================
metrics = [
    ('LA Loss', 'loss'),
    ('Scar Loss', 'loss'),
    ('LA Dice', 'dice'),
    ('Scar Dice', 'dice'),
]

ncols = 2
nrows = 2
pl.figure(figsize=(10, 8))

for i, (metric_name, metric_type) in enumerate(metrics, start=1):
    pl.subplot(nrows, ncols, i)

    ylabel = 'Dice Score' if metric_type == 'dice' else 'Loss'

    # Training curve
    if metric_name in train_data:
        train_epoch = train_data[metric_name]['epoch']
        train_value = train_data[metric_name]['value']

        if metric_type == 'loss':
            valid_train = np.isfinite(train_value) & (train_value > 0)
            pl.plot(
                train_epoch[valid_train],
                np.log(train_value[valid_train]),
                'b-',
                label=f'Train {metric_name}'
            )
            ylabel = 'Loss (log)'
        else:
            pl.plot(
                train_epoch,
                train_value,
                'b-',
                label=f'Train {metric_name}'
            )

    # Validation curve
    val_epoch = val_data[metric_name]['epoch']
    val_value = val_data[metric_name]['value']

    if len(val_epoch) > 0:
        if metric_type == 'loss':
            valid_val = np.isfinite(val_value) & (val_value > 0)
            pl.plot(
                val_epoch[valid_val],
                np.log(val_value[valid_val]),
                'r--',
                label=f'Validation {metric_name}'
            )
            ylabel = 'Loss (log)'
        else:
            pl.plot(
                val_epoch,
                val_value,
                'r--',
                label=f'Validation {metric_name}'
            )

    pl.xlabel('Epoch')
    pl.ylabel(ylabel)
    pl.title(metric_name)

    if metric_type == 'dice':
        pl.ylim(0.0, 1.0)

    pl.legend(frameon=False)

pl.tight_layout()
pl.savefig("img.jpg", dpi=300)
pl.show()