import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================
# Configuration
# ============================================

MODEL_PATH = "models/fruit_quality_finetuned.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = (224, 224)


# ============================================
# Check Command-Line Argument
# ============================================

if len(sys.argv) < 2:
    print("\nUsage:")
    print('python test_model.py "path_to_image"')
    sys.exit(1)


image_path = sys.argv[1]


# ============================================
# Load Model
# ============================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================
# Load Class Names
# ============================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:
    class_names = json.load(file)


# ============================================
# Load Image
# ============================================

image = Image.open(
    image_path
).convert("RGB")

image = image.resize(
    IMG_SIZE
)

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================
# Prediction
# ============================================

predictions = model.predict(
    image_array,
    verbose=0
)[0]


# ============================================
# Get Top 5 Predictions
# ============================================

top_indices = np.argsort(
    predictions
)[::-1][:5]


# ============================================
# Display Results
# ============================================

print("\n" + "=" * 50)
print("Prediction Result")
print("=" * 50)

print(
    f"Top prediction : "
    f"{class_names[top_indices[0]]}"
)

print(
    f"Confidence     : "
    f"{predictions[top_indices[0]] * 100:.2f}%"
)

print("\nTop 5 predictions:")

for rank, index in enumerate(
    top_indices,
    start=1
):
    print(
        f"{rank}. "
        f"{class_names[index]} "
        f"→ "
        f"{predictions[index] * 100:.2f}%"
    )

print("=" * 50)