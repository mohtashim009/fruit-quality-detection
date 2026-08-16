# Fruit Quality Detection

An AI-powered Flask web application that identifies a fruit and classifies its quality as **Fresh** or **Rotten** from an uploaded image.

## Model

The application uses a fine-tuned **MobileNetV2** model trained on a custom fruit-quality dataset with **28 classes**:

- 14 fruit types
- 2 quality categories per fruit: `fresh` and `rotten`

The final fine-tuning run achieved **97.18% validation accuracy**.

The trained model is stored at:

```text
models/fruit_quality_finetuned.keras
```

Class labels are stored in:

```text
models/class_names.json
```

## Project structure

```text
fruit_quality_detection/
├── app.py
├── fruit_model.py
├── nutrient_data.py
├── train.py
├── fine_tune.py
├── evaluate_model.py
├── diagnose_model.py
├── test_model.py
├── requirements.txt
├── models/
│   ├── fruit_quality_model.keras
│   ├── fruit_quality_finetuned.keras
│   └── class_names.json
├── templates/
│   ├── index.html
│   └── result.html
└── uploads/
```

## How the application works

1. The user uploads a fruit image through the Flask interface.
2. The image is converted to RGB and resized to 224×224 pixels.
3. The fine-tuned MobileNetV2 model predicts one of the 28 fruit-quality classes.
4. The class label is split into fruit type and quality (`fresh` or `rotten`).
5. The application displays the predicted fruit, quality, confidence score, uploaded image, and available nutritional information.

## Run locally

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start Flask:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

The fine-tuned model contains its MobileNetV2 preprocessing inside the saved Keras model, so `fruit_model.py` does not apply an additional `/255` normalization step before inference.

Uploaded test images are ignored by Git. Model files are also kept out of the repository if they are covered by the project's `.gitignore`.
