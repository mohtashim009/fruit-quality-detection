import os
import tensorflow as tf

# ==============================
# Configuration
# ==============================

DATASET_DIR = r"G:\Unified_Dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_quality_model.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
VALIDATION_SPLIT = 0.2
SEED = 42


# ==============================
# Check Dataset
# ==============================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_DIR}"
    )

print("Dataset found:", DATASET_DIR)


# ==============================
# Load Dataset
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)


# ==============================
# Class Names
# ==============================

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print("\nTotal classes:", num_classes)


# ==============================
# Improve Dataset Performance
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==============================
# Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")


# ==============================
# Pretrained MobileNetV2
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


# ==============================
# Build Our Classifier
# ==============================

inputs = tf.keras.Input(
    shape=IMG_SIZE + (3,)
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ==============================
# Compile
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# Model Summary
# ==============================

model.summary()


# ==============================
# Train Classifier
# ==============================

print("\nStarting initial training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS
)


# ==============================
# Save Model
# ==============================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)

print("\n====================================")
print("Training completed!")
print("Model saved to:")
print(MODEL_PATH)
print("====================================")