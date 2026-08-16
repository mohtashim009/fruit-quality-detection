import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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

with open(CLASS_NAMES_PATH, "r") as file:
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

                image_paths.append(str(image_path))
                labels.append(class_to_index[label])


image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================
# Recreate Validation Split
# ============================================

_, val_paths, _, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


print(f"\nValidation images: {len(val_paths)}")


# ============================================
# Image Loading
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
# Predictions
# ============================================

print("\nGenerating predictions...")

predictions = model.predict(
    val_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================
# Classification Report
# ============================================

print("\n")
print("=" * 75)
print("PER-CLASS PERFORMANCE")
print("=" * 75)

report = classification_report(
    val_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================
# Confusion Matrix
# ============================================

matrix = confusion_matrix(
    val_labels,
    predicted_labels
)


print("=" * 75)
print("MOST COMMON MISCLASSIFICATIONS")
print("=" * 75)

errors = []

for true_index in range(len(class_names)):

    for predicted_index in range(len(class_names)):

        if true_index == predicted_index:
            continue

        count = matrix[
            true_index,
            predicted_index
        ]

        if count > 0:

            errors.append(
                (
                    count,
                    class_names[true_index],
                    class_names[predicted_index]
                )
            )


errors.sort(
    reverse=True
)


for count, true_class, predicted_class in errors[:20]:

    print(
        f"{count:4d} images: "
        f"{true_class} → {predicted_class}"
    )


# ============================================
# Specific Apple Analysis
# ============================================

print("\n")
print("=" * 75)
print("APPLE ANALYSIS")
print("=" * 75)

for class_name in [
    "apple_fresh",
    "apple_rotten"
]:

    index = class_to_index[class_name]

    total = np.sum(
        val_labels == index
    )

    correct = np.sum(
        (val_labels == index)
        &
        (predicted_labels == index)
    )

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{correct}/{total} correct "
        f"({accuracy * 100:.2f}%)"
    )


# ============================================
# Find Our Specific Apple
# ============================================

target_name = "FreshApple (18).jpg"

matches = [
    path
    for path in val_paths
    if Path(path).name == target_name
]

print("\n")
print("=" * 75)
print("SPECIFIC APPLE IMAGE")
print("=" * 75)

if matches:

    target_path = matches[0]

    target_position = np.where(
        val_paths == target_path
    )[0][0]

    true_index = val_labels[
        target_position
    ]

    predicted_index = predicted_labels[
        target_position
    ]

    confidence = predictions[
        target_position,
        predicted_index
    ] * 100

    print(
        f"Image: {target_path}"
    )

    print(
        f"Actual label: "
        f"{class_names[true_index]}"
    )

    print(
        f"Predicted: "
        f"{class_names[predicted_index]}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

else:

    print(
        "FreshApple (18).jpg is not "
        "in the validation split."
    )

print("=" * 75)