import os
import time
import cv2
import torch
from pathlib import Path

from core.model_loader import load_realesrgan_model

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

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

        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")

        print("Input shape:", img.shape)
        print("Starting inference...")

        start_time = time.time()
        output, _ = self.upsampler.enhance(img, outscale=4)
        end_time = time.time()

        print("Output shape:", output.shape)
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)

        print("Upscaled image saved as:", output_path)

    def upscale_folder(self, input_dir: str, output_dir: str, game_mode: bool = False):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        print(f"Game Mode: {'ON' if game_mode else 'OFF'}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_dir}")

        image_files = [
            file for file in input_path.rglob("*")
            if file.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not image_files:
            print("No supported images found.")
            return

        print(f"Found {len(image_files)} images.")
        processed_count = 0
        skipped_count = 0
        error_count = 0
        start_batch_time = time.time()

        for idx, img_path in enumerate(image_files):
            try:
                print(f"[{idx+1}/{len(image_files)}] Processing {img_path.name}")

                # Phase 3.2 – Skip already upscaled files
                if game_mode:
                    name_lower = img_path.stem.lower()
                    if "_4x" in name_lower or "upscaled" in name_lower:
                        print("Skipped (Already Upscaled)")
                        skipped_count += 1
                        continue
                # Phase 3.3 – Skip small UI / icon images
                if game_mode:
                    img_temp = cv2.imread(str(img_path))
                    if img_temp is None:
                        print("Skipped (Unreadable Image)")
                        continue

                    height, width = img_temp.shape[:2]

                    if min(height, width) < 300:
                        print(f"Skipped (Small Image: {width}x{height})")
                        skipped_count += 1
                        continue
                relative_path = img_path.relative_to(input_path)

                if game_mode:
                    new_name = relative_path.stem + "_4x" + relative_path.suffix
                    save_path = output_path / relative_path.parent / new_name
                else:
                    save_path = output_path / relative_path
                
                self.upscale_image(str(img_path), str(save_path))
                processed_count += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                error_count += 1
        
        end_batch_time = time.time()

        print("\n========== GAME MODE SUMMARY ==========")
        print(f"Total Found: {len(image_files)}")
        print(f"Processed: {processed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        print(f"Total Time: {end_batch_time - start_batch_time:.2f} seconds")
        print("=======================================\n")