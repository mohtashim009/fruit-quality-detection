# Fruit Identification Web App

A full-stack Flask web app that identifies a fruit from an uploaded image and
displays a quality estimate and nutritional information.

## How it works

- **`fruit_model.py`** — loads a pre-trained MobileNetV2 model (ImageNet
  weights) via TensorFlow/Keras. The uploaded image is resized to 224×224,
  normalized, and passed through the model. The top-1 decoded prediction is
  used as the fruit name, and a simple confidence threshold (`> 0.7`) is
  used as a "Good" / "Average" quality heuristic.
- **`nutrient_data.py`** — a small lookup dictionary mapping fruit names to
  basic nutritional info (calories, fiber, vitamin C, etc.).
- **`app.py`** — the Flask app. Handles the image upload, calls the model,
  looks up nutrition info, and renders the result.
- **`templates/`** — `index.html` (upload form) and `result.html`
  (prediction + nutrition display), rendered with Jinja2.

## Tech stack

Python, Flask, TensorFlow/Keras, NumPy, Pillow (PIL)

## Setup

1. Clone the repo and move into it:
   ```bash
   git clone https://github.com/mohtashim009/fruit-quality-detection.git
   cd fruit-quality-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser, upload a fruit image, and
   view the prediction.

## Notes

This project uses the pre-trained MobileNetV2 model directly (ImageNet
weights) for inference — it is not fine-tuned or retrained on a custom
fruit dataset. The "quality" label is a heuristic based on the model's
prediction confidence, not a learned freshness/quality classifier.

## Possible improvements

- Fine-tune MobileNetV2 (or train a smaller custom head) on a labeled
  fruit-quality dataset (fresh vs. rotten) for a real quality classifier
- Add OpenCV-based preprocessing (blur detection, background removal)
- Expand the nutrient database or pull it from a live API
- Add proper error handling and file-type validation on upload
