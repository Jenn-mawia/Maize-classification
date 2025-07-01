from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import io

app = FastAPI()
model = tf.keras.models.load_model("maize_model2.h5")
label_encoder = joblib.load("label_encoder.pkl")

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)
    preds = model.predict(input_tensor)
    label = label_encoder.inverse_transform([np.argmax(preds)])[0]
    confidence = float(np.max(preds))
    return {"label": label, "confidence": round(confidence, 3)}