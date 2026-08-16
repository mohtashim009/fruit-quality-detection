import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(r"G:\Unified_Dataset")

BASE_MODEL_PATH = "models/fruit_quality_model.keras"
FINETUNED_MODEL_PATH = "models/fruit_quality_finetuned.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Fine-tuning configuration
FINE_TUNE_EPOCHS = 3
LEARNING_RATE = 1e-5

# Number of MobileNetV2 layers to unfreeze from the top
UNFREEZE_LAYERS = 30


# ============================================================
# Reproducibility
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Load Class Names
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


# ============================================================
# Collect Dataset
# ============================================================

print("=" * 60)
print("Fruit Quality Detection - Fine-Tuning")
print("=" * 60)

print(f"\nDataset: {DATASET_DIR}")

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

        class_name = f"{fruit_name}_{quality}"

        if class_name not in class_to_index:
            print(
                f"Warning: {class_name} "
                f"not found in class_names.json"
            )
            continue

        class_index = class_to_index[class_name]

        for image_path in quality_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in valid_extensions
            ):

                image_paths.append(
                    str(image_path)
                )

                labels.append(
                    class_index
                )


image_paths = np.array(image_paths)
labels = np.array(labels)


print(f"\nTotal images: {len(image_paths)}")
print(f"Total classes: {len(class_names)}")


# ============================================================
# Dataset Split
# ============================================================

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)


print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")


# ============================================================
# Image Loading
# ============================================================

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


# ============================================================
# Create Training Dataset
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_dataset = train_dataset.shuffle(
    buffer_size=2000,
    seed=SEED
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Create Validation Dataset
# ============================================================

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


# ============================================================
# Load Existing Model
# ============================================================

print("\nLoading existing model...")

model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)


# ============================================================
# Find MobileNetV2 Backbone
# ============================================================

base_model = None

for layer in model.layers:

    if "mobilenetv2" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    raise RuntimeError(
        "Could not find MobileNetV2 inside the model."
    )


print(
    f"Found backbone: {base_model.name}"
)

print(
    f"Total backbone layers: "
    f"{len(base_model.layers)}"
)


# ============================================================
# Freeze Entire Backbone First
# ============================================================

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False


# ============================================================
# Unfreeze Top Layers
# ============================================================

layers_to_unfreeze = base_model.layers[
    -UNFREEZE_LAYERS:
]


for layer in layers_to_unfreeze:

    # Keep BatchNormalization frozen.
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

    else:
        layer.trainable = True


print(
    f"\nUnfreezing last "
    f"{UNFREEZE_LAYERS} backbone layers."
)


# ============================================================
# Display Trainable Information
# ============================================================

trainable_count = 0
non_trainable_count = 0

for layer in model.layers:

    if layer.trainable:
        trainable_count += 1
    else:
        non_trainable_count += 1


print(
    f"Trainable top-level layers: "
    f"{trainable_count}"
)

print(
    f"Frozen top-level layers: "
    f"{non_trainable_count}"
)


# ============================================================
# Recompile With Small Learning Rate
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# Fine-Tuning
# ============================================================

print("\n" + "=" * 60)
print("Starting fine-tuning...")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)


# ============================================================
# Final Evaluation
# ============================================================

print("\n" + "=" * 60)
print("Fine-Tuning Completed")
print("=" * 60)

results = model.evaluate(
    val_dataset,
    verbose=1
)

print(
    f"\nFinal validation loss: "
    f"{results[0]:.4f}"
)

print(
    f"Final validation accuracy: "
    f"{results[1] * 100:.2f}%"
)


# ============================================================
# Save Final Model
# ============================================================

model.save(
    FINETUNED_MODEL_PATH
)


# ============================================================
# Save Class Names
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print(
    f"\nFine-tuned model saved to:"
)

print(
    FINETUNED_MODEL_PATH
)

print("=" * 60)