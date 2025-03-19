import base64
import gc
import io

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__, template_folder="templates")

tf.keras.backend.clear_session()

model = tf.keras.models.load_model("model.keras")

decoder = (
    [str(i) for i in range(10)]
    + [chr(i) for i in range(65, 91)]
    + [chr(i) for i in range(97, 123)]
)


def preprocess_image(image_data):
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(128, 128, 1)
    return np.expand_dims(image_array, axis=0)


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
        print("Shape of Predictions:", predictions.shape)

        predicted_index = np.argmax(predictions)

        if predicted_index >= len(decoder):
            return jsonify({"error": "Index out of bounds for decoder"}), 500

        predicted_character = decoder[predicted_index]

        print(
            f"Predicted Index: {predicted_index}, Predicted Character: {predicted_character}"
        )

        return jsonify({"prediction": predicted_character})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
