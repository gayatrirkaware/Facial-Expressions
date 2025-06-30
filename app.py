from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from datetime import datetime

# Flask app
app = Flask(__name__)

# Load trained 7-class facial classification model
model = load_model("model/facial_model.h5")

# Define the class labels (in the same order used during training)
class_labels = [
    "angry", "disgust", "fear", "happy", "neutral",
    "sad", "surprise"
]

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["facial_expression_classification"]
collection = db["predictions"]

# Image preprocessing
def preprocess_image(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_array = preprocess_image(file)

    # Predict class probabilities
    prediction_probs = model.predict(img_array)[0]  # shape: (15,)
    predicted_index = np.argmax(prediction_probs)
    predicted_label = class_labels[predicted_index]
    predicted_confidence = float(prediction_probs[predicted_index])

    # Save to MongoDB
    collection.insert_one({
        "label": predicted_label,
        "probability": predicted_confidence,
        "timestamp": datetime.now()
    })

    print("Prediction probabilities:", prediction_probs)
    print("Predicted index:", predicted_index)
    print("Predicted label:", predicted_label)
    print("Confidence:", predicted_confidence)


    return jsonify({
        "label": predicted_label,
        "probability": predicted_confidence,
        "all_probabilities": dict(zip(class_labels, map(float, prediction_probs)))
    })



@app.route("/records")
def records():
    data = list(collection.find({}, {"_id": 0}))
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
