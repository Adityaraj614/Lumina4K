import base64
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np
import requests
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, ImageEnhance
from werkzeug.utils import secure_filename

from core.game_mode_engine import GameModeEngine
from core.pipeline import ImagePipeline


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
FILTER_OUTPUTS_DIR = OUTPUTS_DIR / "filter"
BATCH_UPLOADS_DIR = UPLOADS_DIR / "batch"
BATCH_OUTPUTS_DIR = OUTPUTS_DIR / "batch"
ALLOWED_SCALES = {2, 4, 8}
BATCH_SCALE = 4
MAX_BATCH_OUTPUT_PIXELS = 20_000_000
MAX_UPSCALE_OUTPUT_PIXELS = 20_000_000
STYLE_MAP = {
    "mosaic": "mosaic",
    "candy": "candy",
    "rain_princess": "rain_princess",
    "udnie": "udnie"
}
MODEL_PATHS = {
    "mosaic": BASE_DIR / "models" / "mosaic.pth",
    "candy": BASE_DIR / "models" / "candy.pth",
    "rain_princess": BASE_DIR / "models" / "rain_princess.pth",
    "udnie": BASE_DIR / "models" / "udnie.pth"
}
STYLE_MODELS = {}


app = Flask(__name__)


@app.before_request
def auto_cleanup():
    cleanup_old_files(UPLOADS_DIR)
    cleanup_old_files(OUTPUTS_DIR)
    cleanup_batch_directories(BATCH_UPLOADS_DIR)
    cleanup_batch_directories(BATCH_OUTPUTS_DIR)


def ensure_directories() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FILTER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def unique_filename(filename: str) -> str:
    safe_name = secure_filename(filename) or "image"
    stem = Path(safe_name).stem or "image"
    suffix = Path(safe_name).suffix or ".png"
    return f"{stem}_{uuid4().hex}{suffix}"


def get_upscale_error_message(error: Exception) -> str:
    error_text = str(error).lower()

    if any(keyword in error_text for keyword in ("unable to allocate", "out of memory", "std::bad_alloc", "insufficient memory")):
        return "Image too large for selected scale. Please try a smaller scale."

    if any(keyword in error_text for keyword in ("failed to read", "unreadable", "decode")):
        return "The uploaded image could not be processed. Please try a different image file."

    if any(keyword in error_text for keyword in ("failed to save", "write")):
        return "The processed image could not be saved. Please try again."

    return "Image processing failed. Please try a smaller scale or a different image."


def resolve_batch_path(batch_id: str, kind: str, filename: str) -> Path:
    base_dir = BATCH_UPLOADS_DIR if kind == "input" else BATCH_OUTPUTS_DIR
    target_path = (base_dir / batch_id / filename).resolve()
    expected_parent = (base_dir / batch_id).resolve()

    if target_path.parent != expected_parent or not target_path.exists():
        raise FileNotFoundError("Batch asset not found.")

    return target_path


def resolve_filter_output_path(filename: str) -> Path:
    target_path = (FILTER_OUTPUTS_DIR / filename).resolve()
    expected_root = FILTER_OUTPUTS_DIR.resolve()

    if target_path.parent != expected_root or not target_path.exists():
        raise FileNotFoundError("Filtered asset not found.")

    return target_path


def cleanup_old_files(directory: Path, max_age_minutes: int = 30) -> None:
    now = time.time()

    for file in directory.glob("**/*"):
        if file.is_file():
            file_age = now - file.stat().st_mtime

            if file_age > max_age_minutes * 60:
                try:
                    file.unlink()
                except Exception as exc:
                    print(f"Cleanup failed for {file}: {exc}")


def cleanup_batch_directories(base_dir: Path, max_age_minutes: int = 30) -> None:
    now = time.time()

    for folder in base_dir.iterdir():
        if folder.is_dir():
            folder_age = now - folder.stat().st_mtime

            if folder_age > max_age_minutes * 60:
                try:
                    shutil.rmtree(folder)
                except Exception as exc:
                    print(f"Failed to delete batch folder {folder}: {exc}")


def apply_named_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    normalized_filter = (filter_name or "").strip().lower().replace("_", " ").replace("-", " ")
    filtered = image.copy()

    if normalized_filter == "grayscale":
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif normalized_filter == "sepia":
        kernel = np.array(
            [
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189],
            ],
            dtype=np.float32,
        )
        filtered = cv2.transform(filtered, kernel)
    elif normalized_filter == "invert":
        filtered = cv2.bitwise_not(filtered)
    elif normalized_filter == "blur":
        filtered = cv2.GaussianBlur(filtered, (15, 15), 0)
    elif normalized_filter == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        filtered = cv2.filter2D(filtered, -1, kernel)
    elif normalized_filter == "high contrast":
        filtered = cv2.convertScaleAbs(filtered, alpha=1.5, beta=0)
    elif normalized_filter == "brightness boost":
        filtered = cv2.convertScaleAbs(filtered, alpha=1.0, beta=28)
    elif normalized_filter == "warm tone":
        blue, green, red = cv2.split(filtered)
        red = cv2.add(red, 25)
        green = cv2.add(green, 10)
        filtered = cv2.merge((blue, green, red))

    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_image_adjustments(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    sharpness: float,
) -> np.ndarray:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def process_filter_image(
    image_bytes: bytes,
    filter_name: str,
    brightness: float,
    contrast: float,
    saturation: float,
    sharpness: float,
) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded image could not be processed.")

    filtered = apply_named_filter(image, filter_name)
    adjusted = apply_image_adjustments(filtered, brightness, contrast, saturation, sharpness)
    return adjusted


ensure_directories()
pipeline = ImagePipeline()
game_mode_engine = GameModeEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/filters")
def filters():
    return render_template("filters.html")


@app.route("/style-fusion")
def style_fusion():
    return render_template("style_fusion.html")


@app.route("/generate")
def generate():
    return render_template("generate.html")


@app.route("/upscale")
def upscale_page():
    return render_template("upscale.html")


@app.route("/style-transfer", methods=["POST"])
def style_transfer():
    import torch
    from torchvision import transforms
    from PIL import Image, ImageEnhance

    content_file = request.files.get("content")
    style_name = request.form.get("style", "mosaic").strip() or "mosaic"
    strength = float(request.form.get("strength", 0.7))
    strength = max(0.0, min(1.0, strength))

    if not content_file:
        return jsonify({"error": "A content image is required."}), 400

    if style_name not in STYLE_MAP:
        return jsonify({"error": "Unsupported style selected."}), 400

    print("Style selected:", style_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_img = Image.open(content_file).convert("RGB")

    content = transform(content_img).unsqueeze(0).to(device)

    style_key = STYLE_MAP.get(style_name, "mosaic")

    if style_key not in STYLE_MODELS:
        try:
            from models.transformer_net import TransformerNet

            model_path = MODEL_PATHS[style_key]
            if not model_path.exists():
                raise FileNotFoundError(f"Missing style weights: {model_path}")

            model = TransformerNet()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            STYLE_MODELS[style_key] = model
        except Exception as e:
            print("Model load failed:", e)
            return jsonify({"error": f"Failed to load style model: {style_key}"}), 500

    model = STYLE_MODELS[style_key]
    print("Using model:", style_key)

    try:
        with torch.no_grad():
            output = model(content).cpu()
    except Exception as e:
        print("Inference failed:", e)
        return jsonify({"error": str(e)}), 500

    output_img = output.squeeze().clamp(0, 255) / 255
    output_img = transforms.ToPILImage()(output_img)

    styled_np = np.array(output_img).astype(np.float32)
    content_np = np.array(content_img.resize(output_img.size)).astype(np.float32)
    final_np = strength * styled_np + (1 - strength) * content_np
    final_np = np.clip(final_np, 0, 255).astype(np.uint8)
    output_img = Image.fromarray(final_np)

    temp_input_path = OUTPUTS_DIR / f"styled_{uuid4().hex}.png"
    temp_output_path = OUTPUTS_DIR / f"styled_upscaled_{uuid4().hex}.png"

    output_img.save(temp_input_path)

    final_path = temp_input_path
    if output_img.size[0] < 512:
        pipeline.process(
            input_path=str(temp_input_path),
            output_path=str(temp_output_path),
            scale=2
        )
        final_path = temp_output_path

    output_img = Image.open(final_path).convert("RGB")
    output_img = ImageEnhance.Contrast(output_img).enhance(1.2)
    output_img = ImageEnhance.Color(output_img).enhance(1.1)
    output_img = ImageEnhance.Sharpness(output_img).enhance(1.1)
    output_img = ImageEnhance.Brightness(output_img).enhance(1.05)
    output_img.save(final_path)

    return send_file(final_path, mimetype="image/png")


@app.route("/generate-image", methods=["POST"])
def generate_image():
    import replicate
    from PIL import Image

    prompt = request.form.get("prompt", "").strip()
    reference_file = request.files.get("image")
    model_name = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

    if not prompt:
        return jsonify({"error": "A prompt is required."}), 400

    if not os.getenv("REPLICATE_API_TOKEN"):
        return jsonify({"error": "Missing REPLICATE_API_TOKEN environment variable."}), 500

    temp_reference_path = None
    input_payload = None

    try:
        if reference_file and reference_file.filename:
            temp_reference_path = UPLOADS_DIR / unique_filename(reference_file.filename)
            reference_file.save(temp_reference_path)

            with open(temp_reference_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                base64_image = f"data:image/png;base64,{image_base64}"

            input_payload = {
                "prompt": prompt,
                "image": base64_image,
                "prompt_strength": 0.8
            }
        else:
            input_payload = {
                "prompt": prompt
            }

        print("Model:", model_name)
        print("Payload:", input_payload)

        try:
            output = replicate.run(model_name, input=input_payload)
        except Exception as e:
            print("REPLICATE ERROR:", e)
            return jsonify({"error": str(e)}), 500

        if not output:
            return jsonify({"error": "No output from model"}), 500

        image_url = output[0].url if hasattr(output[0], "url") else output[0]

        output_name = f"generated_{uuid4().hex}.png"
        output_path = OUTPUTS_DIR / output_name

        response = requests.get(image_url, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as generated_image:
            generated_image.write(response.content)

        final_path = output_path

        try:
            with Image.open(output_path) as generated_preview:
                if generated_preview.size[0] < 768:
                    upscaled_path = OUTPUTS_DIR / f"generated_upscaled_{uuid4().hex}.png"
                    pipeline.process(
                        input_path=str(output_path),
                        output_path=str(upscaled_path),
                        scale=2,
                    )
                    final_path = upscaled_path
        except Exception:
            final_path = output_path

        return send_file(final_path, mimetype="image/png")
    except Exception as e:
        print("FULL ERROR:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_reference_path and temp_reference_path.exists():
            temp_reference_path.unlink(missing_ok=True)


@app.route("/filter", methods=["POST"])
def filter_image():
    image = request.files.get("image")
    filter_name = request.form.get("filter", "grayscale").strip() or "grayscale"

    if image is None or image.filename == "":
        return jsonify({"error": "Missing image file."}), 400

    try:
        brightness = float(request.form.get("brightness", 1.0))
        contrast = float(request.form.get("contrast", 1.0))
        saturation = float(request.form.get("saturation", 1.0))
        sharpness = float(request.form.get("sharpness", 1.0))
    except ValueError:
        return jsonify({"error": "Adjustment values must be numeric."}), 400

    brightness = min(max(brightness, 0.2), 2.5)
    contrast = min(max(contrast, 0.2), 2.5)
    saturation = min(max(saturation, 0.0), 2.5)
    sharpness = min(max(sharpness, 0.0), 3.0)

    try:
        processed = process_filter_image(
            image.read(),
            filter_name=filter_name,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            sharpness=sharpness,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print("Filter processing failed:", exc)
        return jsonify({"error": "Image processing failed. Please try again."}), 500

    output_name = f"filter_{uuid4().hex}.png"
    output_path = FILTER_OUTPUTS_DIR / output_name

    if not cv2.imwrite(str(output_path), processed):
        return jsonify({"error": "Failed to save filtered image."}), 500

    return jsonify(
        {
            "success": True,
            "image_url": f"/outputs/filter/{output_name}",
        }
    )


@app.route("/apply-filter", methods=["POST"])
def apply_filter():
    image = request.files.get("image")
    filter_name = request.form.get("filter", "grayscale").strip() or "grayscale"

    if image is None or image.filename == "":
        return jsonify({"error": "Missing image file."}), 400

    try:
        processed = process_filter_image(
            image.read(),
            filter_name=filter_name,
            brightness=float(request.form.get("brightness", 1.0)),
            contrast=float(request.form.get("contrast", 1.0)),
            saturation=float(request.form.get("saturation", 1.0)),
            sharpness=float(request.form.get("sharpness", 1.0)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    success, buffer = cv2.imencode(".png", processed)
    if not success:
        return jsonify({"error": "Failed to encode filtered image."}), 500

    return send_file(BytesIO(buffer.tobytes()), mimetype="image/png")


@app.route("/upscale", methods=["POST"])
def upscale_image():
    image = request.files.get("image")
    scale_raw = request.form.get("scale", "").strip()
    mode = request.form.get("mode", "ai").strip().lower() or "ai"

    if image is None or image.filename == "":
        return jsonify({"error": "Missing image file."}), 400

    try:
        scale = int(scale_raw)
    except ValueError:
        return jsonify({"error": "Scale must be one of: 2, 4, 8."}), 400

    if scale not in ALLOWED_SCALES:
        return jsonify({"error": "Scale must be one of: 2, 4, 8."}), 400

    input_name = unique_filename(image.filename)
    output_name = f"upscaled_{input_name}"

    input_path = UPLOADS_DIR / input_name
    output_path = OUTPUTS_DIR / output_name

    image.save(input_path)

    try:
        source_image = cv2.imread(str(input_path))
        if source_image is None:
            raise ValueError("Failed to read uploaded image.")

        height, width = source_image.shape[:2]
        output_pixels = width * scale * height * scale

        if output_pixels > MAX_UPSCALE_OUTPUT_PIXELS:
            if scale == 8:
                scale = 4
            elif scale == 4:
                scale = 2

        if mode == "resize":
            resized_image = cv2.resize(
                source_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )

            if not cv2.imwrite(str(output_path), resized_image):
                raise ValueError("Failed to save resized image.")

            result_path = str(output_path)
        else:
            result_path = pipeline.process(
                input_path=str(input_path),
                output_path=str(output_path),
                scale=scale,
            )
    except Exception as exc:
        return jsonify({"error": get_upscale_error_message(exc)}), 500

    response = send_file(result_path, as_attachment=True)
    response.headers["X-Adjusted-Scale"] = str(scale)
    return response


@app.route("/batch/file/<batch_id>/<kind>/<filename>")
def batch_file(batch_id: str, kind: str, filename: str):
    if kind not in {"input", "output"}:
        abort(404)

    try:
        file_path = resolve_batch_path(batch_id, kind, filename)
    except FileNotFoundError:
        abort(404)

    return send_file(file_path)


@app.route("/outputs/filter/<filename>")
def filter_output(filename: str):
    try:
        file_path = resolve_filter_output_path(filename)
    except FileNotFoundError:
        abort(404)

    return send_file(file_path)


@app.route("/batch/download/<batch_id>")
def batch_download(batch_id: str):
    zip_path = (BATCH_OUTPUTS_DIR / f"{batch_id}.zip").resolve()
    expected_root = BATCH_OUTPUTS_DIR.resolve()

    if zip_path.parent != expected_root or not zip_path.exists():
        abort(404)

    return send_file(zip_path, as_attachment=True, download_name=f"batch_{batch_id}.zip")


@app.route("/batch", methods=["GET"])
def batch_page():
    return render_template("batch.html")


@app.route("/batch", methods=["POST"])
def batch():
    try:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No image files provided."}), 400

        batch_id = uuid4().hex
        batch_input_dir = BATCH_UPLOADS_DIR / batch_id
        batch_output_dir = BATCH_OUTPUTS_DIR / batch_id

        batch_input_dir.mkdir(parents=True, exist_ok=True)
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = 0
        saved_inputs = []
        for file in files:
            if file and file.filename:
                file_path = batch_input_dir / unique_filename(file.filename)
                file.save(file_path)
                saved_files += 1
                saved_inputs.append(file_path)

        if saved_files == 0:
            return jsonify({"error": "No valid image files provided."}), 400

        processed_files = []
        skipped_files = []

        for input_path in saved_inputs:
            try:
                image = cv2.imread(str(input_path))
                if image is None:
                    skipped_files.append(input_path.name)
                    continue

                height, width = image.shape[:2]
                estimated_output_pixels = width * BATCH_SCALE * height * BATCH_SCALE
                if estimated_output_pixels > MAX_BATCH_OUTPUT_PIXELS:
                    skipped_files.append(input_path.name)
                    continue

                output_path = batch_output_dir / input_path.name
                pipeline.process(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    scale=BATCH_SCALE,
                )
                processed_files.append(
                    {
                        "filename": input_path.name,
                        "original": f"/batch/file/{batch_id}/input/{input_path.name}",
                        "upscaled": f"/batch/file/{batch_id}/output/{output_path.name}",
                        "original_url": f"/batch/file/{batch_id}/input/{input_path.name}",
                        "upscaled_url": f"/batch/file/{batch_id}/output/{output_path.name}",
                        "download_url": f"/batch/file/{batch_id}/output/{output_path.name}",
                    }
                )
            except Exception as exc:
                print("Batch item error:", exc)
                skipped_files.append(input_path.name)

        zip_base_path = BATCH_OUTPUTS_DIR / f"{batch_id}"
        shutil.make_archive(str(zip_base_path), "zip", batch_output_dir)
        print("Returning JSON:", len(processed_files), "files")
        return jsonify(
            {
                "success": True,
                "processed_count": len(processed_files),
                "skipped_count": len(skipped_files),
                "skipped_files": skipped_files,
                "results": processed_files,
                "zip_url": f"/batch/download/{batch_id}",
                "zip_download_url": f"/batch/download/{batch_id}",
            }
        )
    except Exception as e:
        print("Batch error:", e)
        print("CRASH:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
