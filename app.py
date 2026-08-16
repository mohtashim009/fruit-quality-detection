import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# app.py
from flask import Flask, request, render_template
from fruit_model import predict_fruit
from nutrient_data import get_nutrient_info

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save and Predict
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    fruit_name, quality = predict_fruit(file_path)

    # Nutrient Info
    nutrients = get_nutrient_info(fruit_name)

    return render_template('result.html', fruit_name=fruit_name, quality=quality, nutrients=nutrients)


if __name__ == '__main__':
    app.run(debug=True)
