from pathlib import Path
import cv2
import time
import torch

from core.pipeline import ImagePipeline   # ✅ NEW

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


class GameModeEngine:
    def __init__(self, min_size=300):
        print("Initializing Game Mode Engine...")
        self.pipeline = ImagePipeline()   # ✅ USE PIPELINE INSTEAD OF UPSCALER
        self.min_size = min_size

    def process(
        self,
        input_dir: str,
        output_dir: str,
        scale=4,
        selected_filters=None,
        style=None   # ✅ NEW (optional)
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_dir}")

        image_files = [
            file for file in input_path.rglob("*")
            if file.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not image_files:
            print("No supported images found.")
            return

        total = len(image_files)
        processed = 0
        skipped = 0
        errors = 0

        print(f"Found {total} images.\n")

        for idx, img_path in enumerate(image_files):
            try:
                progress_percent = ((idx + 1) / total) * 100
                print(f"[{idx+1}/{total}] ({progress_percent:.1f}%) Processing: {img_path.name}")

                start_time = time.time()

                # Preserve folder structure
                relative_path = img_path.relative_to(input_path)
                save_path = output_path / relative_path

                # Create output directory
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if already exists
                if save_path.exists():
                    print("[SKIPPED] Already Exists\n")
                    skipped += 1
                    continue

                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print("[SKIPPED] Unreadable Image\n")
                    skipped += 1
                    continue

                # Skip small images
                height, width = img.shape[:2]
                if min(height, width) < self.min_size:
                    print(f"[SKIPPED] Small Image ({width}x{height})\n")
                    skipped += 1
                    del img
                    continue

                # 🔥 FULL PIPELINE (Upscale → Filters → Style)
                self.pipeline.process(
                    input_path=str(img_path),
                    output_path=str(save_path),
                    scale=scale,
                    selected_filters=selected_filters,
                    style=style   # can be None
                )

                processed += 1

                elapsed = time.time() - start_time
                print(f"[DONE] Time: {elapsed:.2f}s\n")

                # Free CPU memory
                del img

                # Free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}\n")
                errors += 1

        # Final Summary
        print("\n========== GAME MODE SUMMARY ==========")
        print(f"Total Found : {total}")
        print(f"Processed   : {processed}")
        print(f"Skipped     : {skipped}")
        print(f"Errors      : {errors}")
        print("=======================================\n")