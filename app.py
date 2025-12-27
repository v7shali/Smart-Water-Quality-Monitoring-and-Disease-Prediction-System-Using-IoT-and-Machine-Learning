from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# load model bundle once
bundle = joblib.load("kidney_svm_model.pkl")
model = bundle["model"]
label_y = bundle["label_y"]
encoders = bundle["encoders"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

def preprocess(input_dict):
    data = pd.DataFrame([input_dict])
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    for col in cat_cols:
        le = encoders[col]
        data[col] = data[col].fillna(le.classes_[0])
        data[col] = le.transform(data[col])
    data = data.fillna(0)
    return data

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    X = preprocess(payload)
    pred = model.predict(X)[0]
    label = label_y.inverse_transform([pred])[0]
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
