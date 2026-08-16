# Fruit Quality Detection

An AI-powered Flask web application that identifies a fruit and classifies its quality as **Fresh** or **Rotten** from an uploaded image.

The project uses **transfer learning with MobileNetV2**, followed by fine-tuning on a custom fruit-quality dataset containing 28 classes.

## Features

- Fruit identification from an uploaded image
- Fresh/Rotten quality classification
- 28 fruit-quality classes
- Transfer learning using MobileNetV2
- Fine-tuning of the pretrained backbone
- Confidence score for predictions
- Nutritional information for detected fruits
- Flask-based web application
- Model evaluation with precision, recall, and F1-score
- Per-class error analysis
- Reproducible inference using the saved fine-tuned model

---

## Model

The application uses a fine-tuned **MobileNetV2** model.

The model classifies images into **28 classes**:

- 14 fruit/vegetable types
- 2 quality categories for each type:
  - `fresh`
  - `rotten`

### Classes

```text
apple_fresh
apple_rotten
banana_fresh
banana_rotten
bellpepper_fresh
bellpepper_rotten
carrot_fresh
carrot_rotten
cucumber_fresh
cucumber_rotten
grape_fresh
grape_rotten
guava_fresh
guava_rotten
jujube_fresh
jujube_rotten
mango_fresh
mango_rotten
orange_fresh
orange_rotten
pomegranate_fresh
pomegranate_rotten
potato_fresh
potato_rotten
strawberry_fresh
strawberry_rotten
tomato_fresh
tomato_rotten
````

The final fine-tuned model achieved **97.18% validation accuracy** on 5,859 held-out validation images.

The trained model is stored at:

```text
models/fruit_quality_finetuned.keras
```

Class labels are stored at:

```text
models/class_names.json
```

---

## Dataset

The project uses a custom fruit-quality image dataset containing:

```text
Total images: 29,291
Total classes: 28
Training images: 23,432
Validation images: 5,859
```

The dataset contains fresh and rotten examples for each of the 14 supported fruit/vegetable categories.

The dataset is split into training and validation subsets before model training.

---

## Model Architecture

The project uses **MobileNetV2** as the pretrained convolutional neural network backbone.

The initial transfer-learning architecture consists of:

```text
Input Image
    ↓
Data Augmentation
    ↓
MobileNetV2 Backbone
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer
    ↓
28-Class Softmax Output
```

### Input

```text
224 × 224 × 3
```

The saved model contains the MobileNetV2 preprocessing operation, so the inference code does not apply an additional `/255` normalization step.

---

## Training

The initial model was trained using transfer learning.

The MobileNetV2 backbone was initially kept frozen while a new classification head was trained for the 28 fruit-quality classes.

The training pipeline includes:

* Image resizing to 224 × 224
* Data augmentation
* MobileNetV2 pretrained weights
* Global Average Pooling
* Dropout
* 28-class softmax classification
* Adam optimizer
* Sparse categorical cross-entropy loss

The initial trained model is saved locally during the training workflow as:

```text
models/fruit_quality_model.keras
```

The final application uses the fine-tuned model instead.

---

## Fine-Tuning

After the initial transfer-learning stage, the model was fine-tuned to improve its ability to distinguish between the fruit-quality classes.

The fine-tuning process:

1. Loads the initially trained model.
2. Identifies the MobileNetV2 backbone.
3. Keeps most of the backbone frozen.
4. Unfreezes the last 30 backbone layers.
5. Keeps Batch Normalization layers frozen.
6. Uses a low learning rate of `1e-5`.
7. Trains for additional epochs.
8. Saves the best-performing model based on validation accuracy.

The final fine-tuned model achieved:

```text
Validation Accuracy: 97.18%
Validation Loss:     0.0918
```

The fine-tuned model is saved as:

```text
models/fruit_quality_finetuned.keras
```

---

## Model Evaluation

The final fine-tuned model was independently evaluated on **5,859 validation images** across all 28 classes.

### Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 97.18% |
| Validation Loss     | 0.0918 |
| Macro Precision     | 96.00% |
| Macro Recall        | 95.41% |
| Macro F1-Score      | 95.58% |
| Weighted Precision  | 97.22% |
| Weighted Recall     | 97.18% |
| Weighted F1-Score   | 97.15% |

### Strong-performing classes

Several classes achieved F1-scores above 98%, including:

| Class             | F1-Score |
| ----------------- | -------: |
| guava_fresh       |  100.00% |
| strawberry_fresh  |   99.38% |
| strawberry_rotten |   99.21% |
| banana_fresh      |   99.50% |
| banana_rotten     |   99.38% |
| tomato_fresh      |   99.59% |
| apple_fresh       |   98.99% |
| orange_fresh      |   98.79% |
| mango_fresh       |   97.65% |

### More challenging classes

Some classes were more difficult for the model:

| Class             | F1-Score |
| ----------------- | -------: |
| bellpepper_rotten |   83.57% |
| potato_rotten     |   82.88% |
| tomato_rotten     |   90.98% |
| jujube_rotten     |   89.19% |

### Error Analysis

The most common validation-set misclassifications were:

```text
bellpepper_rotten → tomato_rotten : 12 images
carrot_fresh → carrot_rotten      : 11 images
bellpepper_rotten → bellpepper_fresh : 7 images
potato_rotten → potato_fresh      : 6 images
jujube_rotten → jujube_fresh      : 6 images
```

The `diagnose_model.py` script can be used to reproduce the per-class classification report and error analysis.

---

## Inference

The inference pipeline loads the saved fine-tuned model and class labels.

For an uploaded image:

1. The image is converted to RGB.
2. The image is resized to 224 × 224.
3. The image is passed to the fine-tuned MobileNetV2 model.
4. The model produces probabilities for all 28 classes.
5. The class with the highest probability is selected.
6. The class name is separated into:

   * Fruit type
   * Quality
7. The application displays the prediction and confidence score.

For example:

```text
Input:
Apple image

Prediction:
Fruit: Apple
Quality: Fresh
Confidence: 99.32%
```

---

## Flask Web Application

The project includes a Flask web interface for real-time image inference.

### Application workflow

```text
User uploads image
        ↓
Flask receives image
        ↓
Image validation
        ↓
Image preprocessing
        ↓
Fine-tuned MobileNetV2
        ↓
28-class prediction
        ↓
Fruit + Quality + Confidence
        ↓
Nutritional information
        ↓
Result page
```

The application also displays the uploaded image along with the prediction.

---

## Nutritional Information

The application provides basic nutritional information for supported fruits/vegetables, including:

* Calories
* Fiber
* Vitamin C

The nutritional values are stored in:

```text
nutrient_data.py
```

These values are provided for general informational purposes.

---

## Project Structure

```text
fruit_quality_detection/
│
├── app.py
├── fruit_model.py
├── nutrient_data.py
│
├── train.py
├── fine_tune.py
├── evaluate_model.py
├── diagnose_model.py
├── test_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── class_names.json
│   └── fruit_quality_finetuned.keras
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── uploads/
```

---

## Files and Their Purpose

| File                                   | Purpose                                              |
| -------------------------------------- | ---------------------------------------------------- |
| `app.py`                               | Flask web application and upload/prediction routes   |
| `fruit_model.py`                       | Loads the trained model and performs image inference |
| `nutrient_data.py`                     | Contains nutritional information                     |
| `train.py`                             | Initial transfer-learning training pipeline          |
| `fine_tune.py`                         | Fine-tunes the MobileNetV2 backbone                  |
| `evaluate_model.py`                    | Evaluates the final model on the validation set      |
| `diagnose_model.py`                    | Generates per-class metrics and error analysis       |
| `test_model.py`                        | Tests predictions on individual images               |
| `models/fruit_quality_finetuned.keras` | Final fine-tuned model                               |
| `models/class_names.json`              | Model class labels                                   |
| `templates/index.html`                 | Image upload interface                               |
| `templates/result.html`                | Prediction result interface                          |

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mohtashim009/fruit-quality-detection.git
```

Move into the project directory:

```bash
cd fruit-quality-detection
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the Flask application

```powershell
python app.py
```

### 5. Open the application

Open:

```text
http://127.0.0.1:5000
```

Upload a fruit image and the application will return the predicted fruit, quality, confidence score, and available nutritional information.

---

## Running Model Evaluation

To evaluate the final fine-tuned model:

```powershell
python evaluate_model.py
```

To generate detailed per-class metrics and error analysis:

```powershell
python diagnose_model.py
```

To test an individual image:

```powershell
python test_model.py "path/to/image.jpg"
```

---

## Limitations

Although the model achieves **97.18% validation accuracy**, performance on real-world images can vary.

Factors that can affect predictions include:

* Lighting conditions
* Background complexity
* Image quality
* Camera angle
* Fruit orientation
* Unusual fruit appearance
* Differences between dataset images and real-world images
* Visually similar classes

The validation accuracy should therefore not be interpreted as guaranteed real-world accuracy.

The model also has lower performance on some classes, particularly `bellpepper_rotten` and `potato_rotten`, as shown in the per-class evaluation.

---

## Future Improvements

Potential future improvements include:

* Expanding the dataset with more diverse real-world images
* Improving performance on difficult classes
* Adding a confusion matrix visualization
* Adding confidence thresholds for uncertain predictions
* Supporting additional fruits and vegetables
* Deploying the application to a cloud platform
* Adding an API endpoint for external applications
* Experimenting with stronger modern image-classification architectures
* Adding automated model retraining and monitoring

---

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Scikit-learn
* Flask
* HTML/CSS
* Git/GitHub

---

## Project Summary

This project demonstrates an end-to-end computer vision workflow:

```text
Dataset
   ↓
Data Preparation
   ↓
Transfer Learning
   ↓
MobileNetV2
   ↓
Initial Training
   ↓
Fine-Tuning
   ↓
Model Evaluation
   ↓
Inference Testing
   ↓
Flask Web Application
   ↓
Fruit + Quality + Confidence
```

The final fine-tuned model achieved **97.18% validation accuracy**, with a **95.58% macro F1-score** across the 28 fruit-quality classes.

````
