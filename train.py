import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================
# Configuration
# ============================================

DATASET_DIR = Path(r"G:\Unified_Dataset")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fruit_quality_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Initial training of our new classifier
INITIAL_EPOCHS = 5

VALIDATION_SPLIT = 0.20
SEED = 42


# ============================================
# Check Dataset
# ============================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}"
    )

print("=" * 60)
print("Fruit Quality Detection - Training")
print("=" * 60)
print(f"Dataset: {DATASET_DIR}")


# ============================================
# Find Images and Create Labels
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
    [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]
)

for fruit_folder in fruit_folders:

    fruit_name = fruit_folder.name.lower()

    for quality in ["fresh", "rotten"]:

        quality_folder = fruit_folder / quality

        if not quality_folder.exists():
            print(
                f"Warning: Missing folder: {quality_folder}"
            )
            continue

        for image_path in quality_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in valid_extensions
            ):
                image_paths.append(str(image_path))
                labels.append(
                    f"{fruit_name}_{quality}"
                )


image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================
# Display Dataset Information
# ============================================

class_names = sorted(np.unique(labels).tolist())

print("\nDataset information:")
print(f"Total images: {len(image_paths)}")
print(f"Total classes: {len(class_names)}")

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


if len(class_names) != 28:
    print(
        f"\nWARNING: Expected 28 classes, "
        f"but found {len(class_names)}."
    )


# Convert string labels to integer labels
class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

numeric_labels = np.array([
    class_to_index[label]
    for label in labels
])


# ============================================
# Train / Validation Split
# ============================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    numeric_labels,
    test_size=VALIDATION_SPLIT,
    random_state=SEED,
    stratify=numeric_labels
)


print("\nDataset split:")
print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")


# ============================================
# Image Loading Function
# ============================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

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


# ============================================
# Create TensorFlow Datasets
# ============================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)


train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset = train_dataset.shuffle(
    buffer_size=2000,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================
# Data Augmentation
# ============================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),
], name="data_augmentation")


# ============================================
# MobileNetV2
# ============================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================
# Build Model
# ============================================

inputs = tf.keras.Input(
    shape=IMG_SIZE + (3,)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.2
)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================
# Compile
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# Model Summary
# ============================================

print("\nModel summary:")
model.summary()


# ============================================
# Train
# ============================================

print("\n")
print("=" * 60)
print("Starting initial training")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS
)


# ============================================
# Save Model
# ============================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model.save(
    MODEL_PATH
)


# Save class names for prediction later
with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================
# Final Results
# ============================================

final_train_accuracy = history.history[
    "accuracy"
][-1]

final_val_accuracy = history.history[
    "val_accuracy"
][-1]

print("\n")
print("=" * 60)
print("Training completed")
print("=" * 60)

print(
    f"Final training accuracy: "
    f"{final_train_accuracy:.4f}"
)

print(
    f"Final validation accuracy: "
    f"{final_val_accuracy:.4f}"
)

print(f"\nModel saved to:")
print(MODEL_PATH)

print(f"\nClass names saved to:")
print(CLASS_NAMES_PATH)

print("=" * 60)