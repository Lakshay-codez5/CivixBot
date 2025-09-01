# robust_image_inference.py  (drop into your app; import functions where needed)
import io
import numpy as np
from PIL import Image
from typing import Tuple
import json

with open("labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path

# constants
IMG_SIZE = (224, 224)   # match your trained model input size
MODEL_PATH = Path("model.h5")  # adjust if different

# load model (safe loader)
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
    
def handle_text(user_input: str):
    if "hello" in user_input.lower():
        return "Hi! Upload an image of an issue (pothole, garbage, etc.) and I’ll try to recognize it."
    return "Tell me more or send an image."

# preprocess bytes -> batch numpy array
def preprocess_image_bytes(file_bytes: bytes, target_size: Tuple[int,int]=IMG_SIZE) -> np.ndarray:
    """
    Returns: numpy array shape (1, H, W, 3), dtype float32 normalized to [0,1].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # grayscale fallback
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# prediction wrapper (returns label index & confidence)
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
    model = tf.keras.models.load_model("model.h5")
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    result = predict_from_bytes(model, file_bytes)
    result["label"] = CLASS_NAMES[result["label"]] if isinstance(result["label"], int) else result["label"]
    return f"I think this is **{result['label']}** with confidence {result['confidence']}"

if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        elif msg.lower().endswith(".jpg") or msg.lower().endswith(".png"):
            print("Bot:", handle_image(msg))
        else:
            print("Bot:", handle_text(msg))