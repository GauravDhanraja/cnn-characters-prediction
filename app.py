import base64
import io

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__, template_folder="templates")

model = tf.keras.models.load_model("model.keras")
labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def preprocess_image(image_data):
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 1)
    return image_array


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        processed_image = preprocess_image(image_data)

        predictions = model.predict(processed_image)

        print("Raw Predictions:", predictions)

        predicted_index = np.argmax(predictions)
        predicted_character = labels[predicted_index]

        print(
            f"Predicted Index: {predicted_index}, Predicted Character: {predicted_character}"
        )

        return jsonify({"prediction": predicted_character})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
