"""
Utility script to resize all images in Dataset to (200, 200) RGB.
Converts any RGBA or grayscale images cleanly to RGB and ensures uniform dimensions.
"""

import os
from PIL import Image

TARGET_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")


def resize_all_images(dataset_dir=DATASET_DIR, target_size=TARGET_SIZE):
    splits = ["train", "validation", "Test"]
    total_processed = 0
    total_resized = 0
    errors = 0

    print(f"Starting image resizing to {target_size} RGB in: {dataset_dir}")
    print("-" * 60)

    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_path):
            print(f"Skipping missing split directory: {split_path}")
            continue

        split_count = 0
        split_resized = 0

        for root, _, files in os.walk(split_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_path = os.path.join(root, file)
                    split_count += 1
                    try:
                        with Image.open(img_path) as img:
                            needs_resize = (img.size != target_size)
                            needs_convert = (img.mode != "RGB")

                            if needs_resize or needs_convert:
                                # Convert to RGB (handles RGBA, P, L modes cleanly)
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                if img.size != target_size:
                                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                                # Save back with good quality
                                img.save(img_path, format="JPEG", quality=95)
                                split_resized += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        errors += 1

        print(f"[{split:>10}] Processed: {split_count:>4} images | Resized: {split_resized:>4} images")
        total_processed += split_count
        total_resized += split_resized

    print("-" * 60)
    print(f"Finished! Total images: {total_processed}, Resized: {total_resized}, Errors: {errors}")


if __name__ == "__main__":
    resize_all_images()
