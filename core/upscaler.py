import os
import time
import cv2

from core.model_loader import load_realesrgan_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Upscaler:
    def __init__(self):
        print("Initializing RealESRGAN model...")
        self.upsampler = load_realesrgan_model(BASE_DIR)
        print("Model loaded successfully.")

    def upscale_image(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        print("Input shape:", img.shape)

        print("Starting inference...")
        start_time = time.time()

        output, _ = self.upsampler.enhance(img, outscale=4)

        end_time = time.time()
        print("Output shape:", output.shape)
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

        cv2.imwrite(output_path, output)
        print("Upscaled image saved as:", output_path)

if __name__ == "__main__":
    engine = Upscaler()

    input_path = os.path.join(BASE_DIR, "test_input.jpg")
    output_path = os.path.join(BASE_DIR, "test_output_4x.jpg")

    engine.upscale_image(input_path, output_path)
