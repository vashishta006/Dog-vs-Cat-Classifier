from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("dog_cat_model.keras")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        print("FILE RECEIVED:", file.filename)

        image = Image.open(file).convert("RGB")
        image = image.resize((180, 180))

        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        result = model.predict(image_array, verbose=0)[0][0]
        print("MODEL RESULT:", result)

        if result >= 0.5:
            prediction = "DOG 🐶"
            confidence = result * 100
        else:
            prediction = "CAT 🐱"
            confidence = (1 - result) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)