# robust_image_inference.py
import io
import numpy as np
from PIL import Image
from typing import Tuple
import os
from pathlib import Path
import tensorflow as tf

# reduce TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# constants
IMG_SIZE = (224, 224)  # match your trained model input size
MODEL_PATH = Path("model.h5")  # adjust path if needed

# 🟢 Replace with your real class names (order must match training labels)
CLASS_NAMES = ["cat", "dog", "car", "tree"]  

# -------------------------
# load model
# -------------------------
def load_tf_model(path: Path = MODEL_PATH):
    if not path.exists():
        print(f"[model] file not found: {path}")
        return None
    try:
        model = tf.keras.models.load_model(str(path))
        print(f"[model] loaded: {path}")
        return model
    except Exception as e:
        print("[model] failed to load:", e)
        return None

# -------------------------
# preprocess helper
# -------------------------
def preprocess_image_bytes(file_bytes: bytes, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Convert raw image bytes into a batch (1, H, W, 3) numpy array normalized to [0,1].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr

# -------------------------
# prediction wrapper
# -------------------------
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
