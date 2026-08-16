# fruit_model.py
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Pre-Trained Model (MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')


# Function to Process Image
def process_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize for MobileNetV2
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img


# Predict Fruit Name and Quality
def predict_fruit(image_path):
    img = process_image(image_path)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    # Simplify Output
    fruit_name = decoded_preds[0][1]  # Top-1 Prediction
    print(decoded_preds[0])
    confidence = decoded_preds[0][2]  # Confidence
    quality = "Good" if confidence > 0.7 else "Average"
    return fruit_name, quality
