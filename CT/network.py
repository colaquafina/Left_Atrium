from monai.networks.nets import SwinUNETR
from config import IMG_SIZE, NUM_CLASSES

def get_model():
    model = SwinUNETR(
        img_size=IMG_SIZE,
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=24,
        use_checkpoint=True,
    )
    return model