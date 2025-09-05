import io
import numpy as np
from PIL import Image
from typing import Tuple
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path

# Load class labels
with open("labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = Path("model.h5")

def load_tf_model(path: Path):
    if not path.exists():
        print(f"[model] model file not found at: {path}")
        return None
    try:
        model = tf.keras.models.load_model(str(path))
        print("[model] loaded:", path)
        return model
    except Exception as e:
        print("[model] failed to load model:", e)
        return None

def preprocess_image_bytes(file_bytes: bytes, target_size: Tuple[int,int]=IMG_SIZE) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_from_bytes(model, file_bytes: bytes):
    if model is None:
        raise RuntimeError("TF model is not loaded")
    arr = preprocess_image_bytes(file_bytes)
    preds = model.predict(arr)  # shape (1, num_classes)
    probs = preds[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_label = CLASS_NAMES[top_idx]
    return {"label": top_label, "confidence": round(top_conf, 3)}

def handle_image(image_path: str):
    model = load_tf_model(MODEL_PATH)
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    result = predict_from_bytes(model, file_bytes)
    return f"I think this is **{result['label']}** with confidence {result['confidence']}"

if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        elif msg.lower().endswith((".jpg", ".png",".avif")):
            print("Bot:", handle_image(msg))
        else:
            print("Bot: Hi! Upload an image of a civic issue and I’ll try to recognize it.")
