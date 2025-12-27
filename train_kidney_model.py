import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1) Load dataset (CSV must be in same folder as this .py file)
df = pd.read_csv("kidney_disease.csv")

# 2) Clean target column
df["classification"] = df["classification"].str.strip()
df = df.dropna(subset=["classification"])

# Encode target: ckd / notckd -> 0/1
label_y = LabelEncoder()
y = label_y.fit_transform(df["classification"])

# 3) Prepare feature matrix X
if "id" in df.columns:
    X = df.drop(columns=["id", "classification"])
else:
    X = df.drop(columns=["classification"])

# 4) Separate numeric and categorical columns
num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# 5) Handle missing values and encode
# numeric: fill with mean
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# categorical: fill with mode, then label‑encode each column
encoders = {}
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 6) Train‑test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Build and train SVM model
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 8) Evaluate
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9) Save model and encoders for later use
joblib.dump(
    {
        "model": svm_model,
        "label_y": label_y,
        "encoders": encoders,
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
    },
    "kidney_svm_model.pkl",
)
print("Saved model to kidney_svm_model.pkl")
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
