import os
import torch
import time
import cv2
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define model architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    model_path = os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == 'cuda' else False,
        device=device
    )

    input_path = os.path.join(BASE_DIR, "test_input.jpg")
    output_path = os.path.join(BASE_DIR, "test_output_4x.jpg")

    if not os.path.exists(input_path):
        print("Input image not found.")
        return

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    print("Input shape:", img.shape)


    print("Starting inference...")
    start_time = time.time()

    output, _ = upsampler.enhance(img, outscale=4)
    print("Output shape:", output.shape)


    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")

    cv2.imwrite(output_path, output)
    print("Upscaled image saved as:", output_path)


if __name__ == "__main__":
    main()
