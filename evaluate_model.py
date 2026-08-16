import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================
# Configuration
# ============================================

DATASET_DIR = Path(r"G:\Unified_Dataset")

MODEL_PATH = "models/fruit_quality_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


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


class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


# ============================================
# Collect Dataset
# ============================================

image_paths = []
labels = []

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


fruit_folders = sorted(
    [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)


for fruit_folder in fruit_folders:

    fruit_name = fruit_folder.name.lower()

    for quality in ["fresh", "rotten"]:

        quality_folder = fruit_folder / quality

        if not quality_folder.exists():
            continue

        for image_path in quality_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in valid_extensions
            ):

                label = f"{fruit_name}_{quality}"

                image_paths.append(
                    str(image_path)
                )

                labels.append(
                    class_to_index[label]
                )


image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================
# Same Split Used During Training
# ============================================

_, val_paths, _, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


print(
    f"Validation images: {len(val_paths)}"
)


# ============================================
# Load Images
# ============================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================
# Evaluate
# ============================================

print("\nEvaluating model...")

results = model.evaluate(
    val_dataset,
    verbose=1
)

print("\n========================================")
print("Independent Validation Result")
print("========================================")

print(
    f"Loss     : {results[0]:.4f}"
)

print(
    f"Accuracy : {results[1] * 100:.2f}%"
)

print("========================================")