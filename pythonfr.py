from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import python_multipart
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Only keep this if you are using file uploads in FastAPI

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (replace with your trained model path)
MODEL_PATH = "civic_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None  # Placeholder until trained

ISSUE_CLASSES = ["pothole", "garbage", "streetlight_issue", "other"]

@app.post("/chat")
async def chat(message: str = Form(...)):
    # Simple text-based chatbot response
    return {"reply": f"You said: {message}. Please upload an image for issue detection."}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if model is None:
        return {"reply": "⚠️ AI model not loaded. Please train or provide civic_model.h5"}

    # Save uploaded image
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Preprocess image
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = ISSUE_CLASSES[np.argmax(predictions)]

    return {"reply": f"✅ I detected this as a *{predicted_class}*. Do you want to log this issue?"}
