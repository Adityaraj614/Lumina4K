import os
from core.upscaler import Upscaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    engine = Upscaler()

    input_dir = os.path.join(BASE_DIR, "input_images")
    output_dir = os.path.join(BASE_DIR, "output_images")

    engine.upscale_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
