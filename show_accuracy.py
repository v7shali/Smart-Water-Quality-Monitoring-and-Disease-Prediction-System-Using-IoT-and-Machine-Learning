import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load model
bundle = joblib.load("kidney_svm_model.pkl")
model = bundle["model"]
label_y = bundle["label_y"]
encoders = bundle["encoders"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

# Load and prepare data
df = pd.read_csv("kidney_disease.csv")
df["classification"] = df["classification"].str.strip()
df = df.dropna(subset=["classification"])

y = label_y.transform(df["classification"])
X = df.drop(columns=["id", "classification"])

X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
for col in cat_cols:
    le = encoders[col]
    X[col] = X[col].fillna(le.classes_[0])
    X[col] = le.transform(X[col])
X = X.fillna(0)

# Predict and show accuracy
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print("\n" + "="*50)
print("MODEL ACCURACY")
print("="*50)
print(f"Accuracy: {acc:.2f} ({acc*100:.1f}%)")
print("="*50 + "\n")
print(classification_report(y, y_pred))
