import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

original_data = pd.read_csv("preprocessed_churn_data.csv")
X_orig = original_data.drop(columns=["Churn"])
y_orig = original_data["Churn"]

logs = pd.read_csv("user_logs.csv")

if "actual_churn" not in logs.columns:
    logs["actual_churn"] = logs["prediction"]

# Drop unused columns for features
X_new = logs.drop(columns=["timestamp", "prediction", "probability", "actual_churn"])
y_new = logs["actual_churn"]

# Combine datasets
X_combined = pd.concat([X_orig, X_new], ignore_index=True)
y_combined = pd.concat([y_orig, y_new], ignore_index=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run(run_name="Retrained Churn Model"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(model, "model")

print("Retrained and logged with actual_churn (or fallback to prediction).")
