"""
Inference script for Animal Image Classification.
Predicts animal class (cat, cow, dog, elephant, horse) from an image file or directory.
"""

import os
import argparse
from gpu_config import setup_gpu

# Initialize GPU libraries and memory growth before heavy TF operations
setup_gpu()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


CLASS_NAMES = ["cat", "cow", "dog", "elephant", "horse"]
EMOJI_MAP = {
    "cat": "🐱 Cat",
    "cow": "🐮 Cow",
    "dog": "🐶 Dog",
    "elephant": "🐘 Elephant",
    "horse": "🐴 Horse"
}


def load_classifier_model(model_path=None):
    """Load trained model from disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        candidates = [
            os.path.join(base_dir, "models", "CNN_predict_model.keras"),
            os.path.join(base_dir, "models", "CNN_best_model.keras"),
            os.path.join(base_dir, "models", "CNN_predict_model.h5")
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Looked at: {model_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def predict_image(model, img_path, target_size=(200, 200)):
    """Predict the class of an image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    # Rescale matching training preprocessing (1/255)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    best_idx = np.argmax(predictions)
    best_class = CLASS_NAMES[best_idx]
    confidence = predictions[best_idx] * 100

    return best_class, confidence, predictions


def main():
    parser = argparse.ArgumentParser(description="Predict Animal Class from Image")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory containing images")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    args = parser.parse_args()

    model = load_classifier_model(args.model)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # If neither provided, run on a few sample test images
    if not args.image and not args.dir:
        sample_images = []
        for c in CLASS_NAMES:
            c_dir = os.path.join(base_dir, "Dataset", "Test", c)
            if os.path.isdir(c_dir):
                files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_images.append((c, os.path.join(c_dir, files[0])))

        print(f"\nRunning predictions on sample test images (1 per class):")
        print("=" * 60)
        for true_class, img_path in sample_images:
            pred_class, conf, all_probs = predict_image(model, img_path)
            emoji_str = EMOJI_MAP.get(pred_class, pred_class)
            print(f"Ground Truth: {true_class:<10} | Predicted: {emoji_str:<12} ({conf:5.1f}% confidence)")
        print("=" * 60)
        return

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: image not found at '{args.image}'")
            return
        pred_class, conf, all_probs = predict_image(model, args.image)
        emoji_str = EMOJI_MAP.get(pred_class, pred_class)
        print(f"\nImage:      {args.image}")
        print(f"Prediction: {emoji_str} ({conf:.2f}%)")
        print("\nAll Class Probabilities:")
        for name, prob in zip(CLASS_NAMES, all_probs):
            print(f" - {EMOJI_MAP.get(name, name):<12}: {prob * 100:5.2f}%")

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: directory not found at '{args.dir}'")
            return
        image_files = [
            f for f in os.listdir(args.dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"\nFound {len(image_files)} images in {args.dir}:")
        print("=" * 60)
        for fname in image_files[:20]:
            img_path = os.path.join(args.dir, fname)
            pred_class, conf, _ = predict_image(model, img_path)
            emoji_str = EMOJI_MAP.get(pred_class, pred_class)
            print(f"{fname:<30} -> {emoji_str:<12} ({conf:5.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
