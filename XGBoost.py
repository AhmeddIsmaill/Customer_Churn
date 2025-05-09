import pandas as pd
import numpy as np
import sys
import io
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("XGBoost script execution started...")

# Load the dataset
data = pd.read_csv('preprocessed_churn_data.csv', encoding='utf-8')
print("Loaded columns:", data.columns.tolist())

# Model Selection and Data Preparation
X = data.drop(columns=['Churn'])
y = data['Churn']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Model setup
model_name = "XGBoost"
model = XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# MLflow run
with mlflow.start_run(run_name=model_name):
    mlflow.log_param("train_set_size", X_train.shape[0])
    mlflow.log_param("test_set_size", X_test.shape[0])

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    mlflow.log_metric("cv_f1_mean", scores.mean())
    mlflow.log_metric("cv_f1_std", scores.std() * 2)
    print(f"{model_name} CV F1 Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print(f"Starting GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print(f"{model_name} Best Parameters: {best_params}")
    print(f"{model_name} Best CV F1 Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    evaluation_results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    for metric, value in evaluation_results.items():
        mlflow.log_metric(f"test_{metric.lower()}", value)

    print(f"\n{model_name} Test Set Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Log the model
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Logged {model_name} model to MLflow")

    # Markdown report
    performance_report = """
    # {0} Performance Report

    ## Test Set Metrics
    | Metric     | Value    |
    |------------|----------|
    | Accuracy   | {1:.4f}  |
    | Precision  | {2:.4f}  |
    | Recall     | {3:.4f}  |
    | F1-Score   | {4:.4f}  |
    | ROC-AUC    | {5:.4f}  |

    ## Cross-Validation
    - F1 Score: {6:.4f} (+/- {7:.4f})

    ## Best Parameters
    {8}

    ## Insights
    - Confusion matrix saved as {9}.
    - Check MLflow run for detailed logs.
    """

    report = performance_report.format(
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        scores.mean(),
        scores.std() * 2,
        '\n'.join([f"- {k}: {v}" for k, v in best_params.items()]),
        cm_path
    )

    report_path = f"{model_name.lower().replace(' ', '_')}_performance_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nModel Performance Report saved as '{report_path}'")

print(f"\n{model_name} script completed successfully!")
