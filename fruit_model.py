import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fruit_quality_finetuned.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)


def _load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


print("Loading fine-tuned fruit quality model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = _load_class_names()

print(
    f"Fruit quality model loaded successfully. "
    f"Number of classes: {len(class_names)}"
)


def process_image(image_path):
    """Load and resize an image for the trained MobileNetV2 model.

    The trained model already contains its MobileNetV2 preprocessing layer,
    so the image is kept in the normal 0-255 pixel range here.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


def predict_fruit(image_path):
    """Predict fruit type, quality, and confidence."""
    image = process_image(image_path)
    probabilities = model.predict(image, verbose=0)[0]

    top_index = int(np.argmax(probabilities))
    class_name = class_names[top_index]
    confidence = float(probabilities[top_index]) * 100

    # Classes are named like apple_fresh / apple_rotten.
    if class_name.endswith("_fresh"):
        quality = "Fresh"
        fruit_name = class_name[:-6]
    elif class_name.endswith("_rotten"):
        quality = "Rotten"
        fruit_name = class_name[:-7]
    else:
        fruit_name = class_name
        quality = "Unknown"

    fruit_name = fruit_name.replace("_", " ").title()

    print("=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"Fruit      : {fruit_name}")
    print(f"Quality    : {quality}")
    print(f"Confidence : {confidence:.2f}%")
    print("=" * 50)

    return fruit_name, quality, confidence