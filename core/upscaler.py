from PIL import Image
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_image(path):
    return Image.open(path)

def save_image(image, path):
    image.save(path)

if __name__ == "__main__":
    input_path = os.path.join(BASE_DIR, "test_input.jpg")
    output_path = os.path.join(BASE_DIR, "test_output.jpg")

    if os.path.exists(input_path):
        img = load_image(input_path)
        save_image(img, output_path)
        print("Image loaded and saved successfully.")
    else:
        print("File not found at:", input_path)
