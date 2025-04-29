import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog

def classify_animal():
    # Open file dialog to select image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if not image_path:
        print("No image selected.")
        return

    # Load the trained model
    model = tf.keras.models.load_model('best_model_transfer_learning.keras', compile=False)

    # Preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    result = "Non-Dangerous Animal" if prediction > 0.5 else "Dangerous Animal"
    print(f"Prediction: {result}")

# Call the function
classify_animal()
