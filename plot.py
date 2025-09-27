import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# Load dataset
data = pd.read_csv("C:\\Users\\hp\\Downloads\\waterborne_data.csv")

X = data.drop("Disease", axis=1)
y = data["Disease"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Decode labels
y_pred_labels = le.inverse_transform(y_pred)

# Count predictions
prediction_counts = pd.Series(y_pred_labels).value_counts()
colors = plt.cm.tab10.colors  

# ---------- 2D Bar Chart ----------
plt.figure(figsize=(8,6))
plt.bar(prediction_counts.index, prediction_counts.values, color=colors[:len(prediction_counts)], edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of Predictions")
plt.title("Predicted Waterborne Diseases Distribution (Bar Chart)", fontsize=14, weight="bold")
plt.tight_layout()
plt.show()

# ---------- 2D Pie Chart ----------
plt.figure(figsize=(7,7))
plt.pie(
    prediction_counts.values,
    labels=prediction_counts.index,
    autopct="%1.1f%%",
    colors=colors[:len(prediction_counts)],
    startangle=140
)
plt.title("Predicted Waterborne Diseases Distribution (Pie Chart)", fontsize=14, weight="bold")
plt.show()
