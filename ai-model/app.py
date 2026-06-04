from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import pipeline

app = Flask(__name__)

# Load model
classifier = pipeline("image-classification", model="deepfake_model.H5/deepfake_vs_real_image_detection")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream)

    result = classifier(img)[0]

    label = result['label']
    score = round(result['score'] * 100, 2)

    return jsonify({
        "label": label,
        "confidence": score
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
