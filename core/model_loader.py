import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def load_realesrgan_model(base_dir: str):
    """
    Initializes and returns a configured RealESRGANer model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    model_path = os.path.join(base_dir, "weights", "RealESRGAN_x4plus.pth")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == "cuda" else False,
        device=device
    )

    return upsampler
