import os
import shutil
import cv2

from core.upscaler import Upscaler
from core.style_engine import StyleEngine


class ImagePipeline:
    def __init__(self):
        self.upscaler = Upscaler()
        self.style_engine = StyleEngine()

    def process(
        self,
        input_path,
        output_path,
        scale=None,              # OPTIONAL
        selected_filters=None,   # OPTIONAL
        style=None               # OPTIONAL
    ):
        """
        Flexible Pipeline:

        User can choose:
        - Only upscale
        - Only filters
        - Only style
        - Any combination
        """

        # -------------------------
        # Setup
        # -------------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base, ext = os.path.splitext(output_path)

        current_path = input_path

        temp_upscaled = f"{base}_upscaled{ext}"
        temp_filtered = f"{base}_filtered{ext}"

        # -------------------------
        # Step 1: Upscale (OPTIONAL)
        # -------------------------
        if scale:
            self.upscaler.upscale_image(
                current_path,
                temp_upscaled,
                scale=scale,
                selected_filters=None
            )
            current_path = temp_upscaled

        # -------------------------
        # Step 2: Filters (OPTIONAL)
        # -------------------------
        if selected_filters:
            image = cv2.imread(current_path)

            if image is None:
                raise ValueError(f"[Pipeline] Failed to read image: {current_path}")

            image = self.upscaler.post_process(image, selected_filters)

            cv2.imwrite(temp_filtered, image)
            current_path = temp_filtered

        # -------------------------
        # Step 3: Style (OPTIONAL)
        # -------------------------
        if style:
            final_output = self.style_engine.apply_style(
                input_path=current_path,
                style_name=style,
                output_path=output_path
            )
        else:
            # Safe copy instead of rename
            shutil.copy(current_path, output_path)
            final_output = output_path

        # -------------------------
        # Cleanup temp files
        # -------------------------
        try:
            if os.path.exists(temp_upscaled):
                os.remove(temp_upscaled)

            if os.path.exists(temp_filtered):
                os.remove(temp_filtered)
        except Exception:
            pass  # ignore cleanup errors

        return final_output