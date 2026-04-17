from core.upscaler import Upscaler
from core.game_mode_engine import GameModeEngine
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Lumina4K - AI Image Upscaler")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--game_mode", action="store_true")

    args = parser.parse_args()

    if args.game_mode:
        engine = GameModeEngine()
        engine.process(args.input, args.output)
    else:
        upscaler = Upscaler()
        if os.path.isfile(args.input):
            upscaler.upscale_image(args.input, args.output)
        else:
            upscaler.upscale_folder(args.input, args.output)


if __name__ == "__main__":
    main()