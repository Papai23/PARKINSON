import joblib
from extract_features import extract_features

print("Loading model...")
model = joblib.load("xgboost_model.pkl")

print("Extracting features...")
audio = "voice.wav"
features = extract_features(audio)

print("Features extracted")

prediction = model.predict([features])

print("Prediction done")

if prediction == 1:
    print("Parkinson detected")
else:
    print("Healthy")