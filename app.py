"""
Bone Fracture Detection - Web App (Flask)
Run:  python app.py   ->  http://127.0.0.1:5000
Place the model file (fracture_model.keras) inside the model/ folder.
"""
import io, os, json
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_CANDIDATES = ["fracture_model.keras", "best_model.keras", "fracture_model.h5"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

model = None
model_error = None
CLASS_NAMES = ["fractured", "not fractured"]   # index 0, index 1
IMG_SIZE = (224, 224)


def load_everything():
    global model, model_error, CLASS_NAMES, IMG_SIZE
    cfg = os.path.join(MODEL_DIR, "class_names.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            data = json.load(f)
        CLASS_NAMES = data.get("class_names", CLASS_NAMES)
        IMG_SIZE = tuple(data.get("img_size", IMG_SIZE))

    path = next((os.path.join(MODEL_DIR, n) for n in MODEL_CANDIDATES
                 if os.path.exists(os.path.join(MODEL_DIR, n))), None)
    if path is None:
        model_error = ("Model file not found. Put 'fracture_model.keras' "
                       "inside the 'model' folder and restart the app.")
        print("[!]", model_error)
        return
    try:
        from tensorflow import keras
        model = keras.models.load_model(path, compile=False)
        print(f"[OK] Model loaded: {path}")
    except Exception as e:
        model_error = f"Failed to load model: {e}"
        print("[!]", model_error)


def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")          # 0-255 (preprocessing is built into the model)
    return np.expand_dims(arr, 0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(ready=model is not None, error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error=model_error or "Model is not ready."), 503
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image was received."), 400
    try:
        x = preprocess(file.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="This is not a valid image file (please use JPG/PNG)."), 400

    p_class1 = float(model.predict(x, verbose=0).ravel()[0])  # prob of 'not fractured'
    p_frac = 1.0 - p_class1
    probs = {CLASS_NAMES[0]: p_frac, CLASS_NAMES[1]: p_class1}
    label = CLASS_NAMES[0] if p_frac >= 0.5 else CLASS_NAMES[1]
    return jsonify(label=label,
                   is_fractured=label == CLASS_NAMES[0],
                   confidence=round(max(p_frac, p_class1) * 100, 2),
                   probabilities={k: round(v * 100, 2) for k, v in probs.items()})


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 15 MB."), 413


load_everything()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
