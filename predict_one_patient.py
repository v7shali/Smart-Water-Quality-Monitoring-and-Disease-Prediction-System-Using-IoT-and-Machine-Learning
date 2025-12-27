import joblib
import pandas as pd

# 1) Load the saved model bundle
bundle = joblib.load("kidney_svm_model.pkl")
model = bundle["model"]
label_y = bundle["label_y"]
encoders = bundle["encoders"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

# 2) Example patient data
data = pd.DataFrame([{
    "age": 48.0,
    "bp": 80.0,
    "sg": 1.02,
    "al": 1.0,
    "su": 0.0,
    "rbc": "normal",
    "pc": "normal",
    "pcc": "notpresent",
    "ba": "notpresent",
    "bgr": 121.0,
    "bu": 36.0,
    "sc": 1.2,
    "sod": 135.0,      # give real numeric instead of None
    "pot": 4.5,        # give real numeric instead of None
    "hemo": 15.4,
    "pcv": "44",
    "wc": "7800",
    "rc": "5.2",
    "htn": "yes",
    "dm": "yes",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no",
}])

# 3) Same preprocessing as training
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

for col in cat_cols:
    le = encoders[col]
    data[col] = data[col].fillna(le.classes_[0])
    data[col] = le.transform(data[col])

# Extra safety: ensure NO NaN anywhere
data = data.fillna(0)

# 4) Predict
pred = model.predict(data)[0]
label = label_y.inverse_transform([pred])[0]

print("Predicted class:", label)   # ckd or notckd
