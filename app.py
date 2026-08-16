import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from fruit_model import predict_fruit
from nutrient_data import get_nutrient_info

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.secret_key = "fruit-quality-detection-local"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Please choose an image first.")
        return redirect(url_for("home"))

    file = request.files["file"]

    if not file.filename:
        flash("Please choose an image first.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload JPG, JPEG, PNG, JFIF, or WEBP.")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        fruit_name, quality, confidence = predict_fruit(str(file_path))
        nutrients = get_nutrient_info(fruit_name)

        return render_template(
            "result.html",
            fruit_name=fruit_name,
            quality=quality,
            confidence=confidence,
            nutrients=nutrients,
            image_filename=filename,
        )
    except Exception as exc:
        print(f"Prediction error: {exc}")
        flash("We couldn't process that image. Please try another clear fruit image.")
        return redirect(url_for("home"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_error):
    flash("The image is too large. Please upload an image smaller than 10 MB.")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
