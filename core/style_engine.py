import os
import torch
from PIL import Image
import torchvision.transforms as transforms


# -----------------------------
# Style Model Registry
# -----------------------------
STYLE_MODELS = {
    "anime": "weights/styles/anime.pt",
    "cyberpunk": "weights/styles/cyberpunk.pt",
    "sketch": "weights/styles/sketch.pt",
    "oil": "weights/styles/oil.pt"
}


# -----------------------------
# Style Engine Class
# -----------------------------
class StyleEngine:
    def __init__(self, device="cuda", max_size=720):
        """
        device: "cuda" or "cpu"
        max_size: max dimension for input image (to control VRAM)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.loaded_models = {}  # cache models

    # -----------------------------
    # Load Style Model (Cached)
    # -----------------------------
    def load_model(self, style_name):
        if style_name not in STYLE_MODELS:
            raise ValueError(f"[StyleEngine] Style '{style_name}' not found.")

        # return cached model if already loaded
        if style_name in self.loaded_models:
            return self.loaded_models[style_name]

        model_path = STYLE_MODELS[style_name]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[StyleEngine] Model not found: {model_path}")

        try:
            model = torch.jit.load(model_path, map_location=self.device)  # ✅ FIX
        except Exception:
            # fallback if not TorchScript
            model = torch.load(model_path, map_location=self.device)     # ✅ FIX

        model.to(self.device)
        model.eval()

        self.loaded_models[style_name] = model
        return model

    # -----------------------------
    # Resize (Maintain Aspect Ratio)
    # -----------------------------
    def resize_image(self, image):
        width, height = image.size

        if max(width, height) <= self.max_size:
            return image

        scale = self.max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    # -----------------------------
    # Preprocess Image → Tensor
    # -----------------------------
    def preprocess(self, image):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(self.device)

    # -----------------------------
    # Postprocess Tensor → Image
    # -----------------------------
    def postprocess(self, tensor):
        tensor = tensor.squeeze().cpu().clamp(0, 1)
        return transforms.ToPILImage()(tensor)

    # -----------------------------
    # Apply Style
    # -----------------------------
    def apply_style(self, input_path, style_name, output_path=None):
        """
        input_path: path to image after filters
        style_name: selected style
        output_path: where to save result
        """

        if style_name is None:
            return input_path  # no style applied

        # load model
        model = self.load_model(style_name)

        # load image
        image = Image.open(input_path).convert("RGB")

        # resize (VRAM safety)
        image = self.resize_image(image)

        # preprocess
        input_tensor = self.preprocess(image)

        # inference
        with torch.no_grad():
            try:
                output_tensor = model(input_tensor)
            except Exception as e:
                raise RuntimeError(f"[StyleEngine] Model inference failed: {e}")

        # postprocess
        output_image = self.postprocess(output_tensor)

        # define output path
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_{style_name}{ext}"

        # ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ✅ NEW

        # save
        output_image.save(output_path)

        return output_path

    # -----------------------------
    # List Available Styles
    # -----------------------------
    def list_styles(self):
        return list(STYLE_MODELS.keys())