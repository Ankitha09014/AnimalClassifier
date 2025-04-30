from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model/best_model_transfer_learning.keras', compile=False)

# Load metrics
metrics = {"val_accuracy": "N/A", "val_loss": "N/A"}
try:
    with open("model/metrics.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            metrics[key] = f"{float(value) * 100:.2f}%" if "accuracy" in key else value
except FileNotFoundError:
    print("metrics.txt not found.")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    show_image = False

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = "static/uploaded_image.jpg"
            file.save(file_path)

            try:
                img = image.load_img(file_path, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction_value = model.predict(img_array)[0][0]
                prediction = "Non-Dangerous Animal" if prediction_value > 0.5 else "Dangerous Animal"
                show_image = True

            except Exception as e:
                prediction = f"Error during prediction: {str(e)}"
                show_image = False

    return render_template("index.html", prediction=prediction, metrics=metrics, show_image=show_image)
if __name__ == "__main__":
    app.run(debug=True)
