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

    # =========================
    # 🔥 FILTER FUNCTIONS (KEEP THESE)
    # =========================

    def apply_sharpen(self, image):
        kernel = [[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]]
        kernel = cv2.UMat(kernel)
        return cv2.filter2D(image, -1, kernel)

    def apply_contrast(self, image):
        alpha = 1.2
        beta = 10
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def apply_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def apply_denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def apply_smooth(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    # =========================
    # 🎯 STYLE PRESETS (KEEP THESE)
    # =========================

    STYLE_PRESETS = {
        "cinematic": ["contrast", "sharpen"],
        "black_white": ["grayscale"],
        "studio": ["denoise", "sharpen"],
        "smooth": ["smooth"],
        "sharp": ["sharpen"]
    }

    # =========================
    # 🧠 POST PROCESS (KEEP FOR PIPELINE USE)
    # =========================

    def post_process(self, image, selected_filters=None):
        if not selected_filters:
            return image

        selected_filters = selected_filters[:3]

        actions = []

        for f in selected_filters:
            if f in self.STYLE_PRESETS:
                actions.extend(self.STYLE_PRESETS[f])

        actions = list(set(actions))

        for action in actions:
            if action == "sharpen":
                image = self.apply_sharpen(image)

            elif action == "contrast":
                image = self.apply_contrast(image)

            elif action == "grayscale":
                image = self.apply_grayscale(image)

            elif action == "denoise":
                image = self.apply_denoise(image)

            elif action == "smooth":
                image = self.apply_smooth(image)

        return image

    # =========================
    # 🚀 SINGLE IMAGE UPSCALE (UPDATED)
    # =========================

    def upscale_image(self, input_path: str, output_path: str, scale=4, selected_filters=None):
        """
        ⚠️ IMPORTANT:
        Filters are NO LONGER applied here.
        This function ONLY does upscaling.
        """

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")

        print("Input shape:", img.shape)
        print("Starting inference...")

        start_time = time.time()

        # 🔥 First pass (always 4x)
        output, _ = self.upsampler.enhance(img, outscale=4)

        # 🎯 Handle scale options
        if scale == 2:
            h, w = output.shape[:2]
            output = cv2.resize(output, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        elif scale == 8:
            output, _ = self.upsampler.enhance(output, outscale=4)

        end_time = time.time()

        print("Output shape:", output.shape)
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)

        print("Upscaled image saved as:", output_path)

    # =========================
    # ⚠️ LEGACY FOLDER METHOD (UNCHANGED)
    # =========================

    def upscale_folder(self, input_dir: str, output_dir: str, game_mode: bool = False):
        """
        ⚠️ Legacy method (use GameModeEngine instead)
        """

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

                if game_mode:
                    name_lower = img_path.stem.lower()
                    if "_4x" in name_lower or "upscaled" in name_lower:
                        print("Skipped (Already Upscaled)")
                        skipped_count += 1
                        continue

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