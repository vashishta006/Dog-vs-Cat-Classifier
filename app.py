from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load lightweight TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="dog_cat_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            print("FILE RECEIVED:", file.filename)

            image = Image.open(file).convert("RGB")
            image = image.resize((180, 180))

            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # MobileNetV2 preprocessing
            image_array = (image_array / 127.5) - 1.0

            # Run TFLite model
            interpreter.set_tensor(
                input_details[0]["index"],
                image_array
            )

            interpreter.invoke()

            result = interpreter.get_tensor(
                output_details[0]["index"]
            )[0][0]

            print("MODEL RESULT:", result)

            if result >= 0.5:
                prediction = "DOG 🐶"
                confidence = float(result) * 100
            else:
                prediction = "CAT 🐱"
                confidence = float(1 - result) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run()