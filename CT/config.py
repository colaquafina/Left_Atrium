import os

LABEL_ROOT = "/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/train_data/label_data"
UNLABEL_ROOT = "/work/users/g/s/gsonw/BIOS740/final_project/cardiac_anatomy_segmentation/train_data/Nolabel_data"
OUTPUT_DIR = "/work/users/g/s/gsonw/BIOS740/final_project/CT/outputs"

NUM_CLASSES = 4
IMG_SIZE = (96, 96, 96)
PIXDIM = (1.5, 1.5, 2.0)

A_MIN = 0
A_MAX = 4095
B_MIN = 0.0
B_MAX = 1.0

BATCH_SIZE = 1
MAX_EPOCHS = 400
VAL_INTERVAL = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5

TRAIN_NUM_WORKERS = 4
VAL_NUM_WORKERS = 2
CACHE_RATE = 1.0
MAX_CPS_WEIGHT = 1.0
RAMPUP_EPOCHS = 30
CONFIDENCE_THRESHOLD = 0.95

RANDOM_STATE = 42
DEVICE = "cuda"