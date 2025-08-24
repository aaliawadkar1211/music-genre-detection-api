from fastapi import FastAPI, File, UploadFile
import uvicorn
import joblib
import librosa
import numpy as np
import os
import tempfile
from pydantic import BaseModel

# Load model and scaler
model = joblib.load("genre_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Feature extractor (same as training)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        features = {}

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std']  = np.std(mfccs[i])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma{i+1}_std']  = np.std(chroma[i])

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'contrast{i+1}_mean'] = np.mean(contrast[i])
            features[f'contrast{i+1}_std']  = np.std(contrast[i])

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz{i+1}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz{i+1}_std']  = np.std(tonnetz[i])

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std']  = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std']  = np.std(rms)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Init app
app = FastAPI(title="Music Genre Classifier API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract features
    feats = extract_features(tmp_path)
    os.remove(tmp_path)

    if not feats:
        return {"error": "Feature extraction failed"}

    # Convert to numpy array
    X = np.array(list(feats.values())).reshape(1, -1)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    pred = model.predict(X_scaled)[0]

    return {"predicted_genre": pred}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
