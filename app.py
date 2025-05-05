from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('best_model_transfer_learning.keras', compile=False)

# Define the path to the class labels
class_labels = os.listdir('data')  # This assumes your classes are directly under 'data'

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    show_image = False

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = "static/uploaded_image.jpg"
            file.save(file_path)

            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_labels[predicted_class_index]
            prediction = f"This is a {predicted_class}."

            show_image = True

    return render_template("index.html", prediction=prediction, show_image=show_image)

if __name__ == "__main__":
    app.run(debug=True)