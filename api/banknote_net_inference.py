import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import joblib
import io

# Load encoder and classifier
try:
    encoder = load_model('banknote_net_assets/banknote_net_encoder.h5')
    clf = joblib.load('banknote_net_assets/banknote_classifier.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert('RGB').resize((224, 224))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_banknote(image_bytes):
    arr = preprocess_image(image_bytes)
    embedding = encoder.predict(arr)
    pred = clf.predict(embedding)
    proba = clf.predict_proba(embedding)
    confidence = np.max(proba)
    return pred[0], confidence
