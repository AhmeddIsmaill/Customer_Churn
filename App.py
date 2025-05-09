from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
import uvicorn
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up logging
logging.basicConfig(
    filename="prediction_logs.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Customer Churn Prediction API")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def get_best_model_path():
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.`test_f1-score` DESC"],
            max_results=1
        )
        if runs:
            run_id = runs[0].info.run_id
            return f"mlartifacts/0/{run_id}/artifacts/model"
        else:
            raise Exception("No completed runs with metric 'test_f1-score' found.")
    except Exception as e:
        logging.error(f"Failed to fetch best model: {e}")
        raise RuntimeError("Could not fetch best model from MLflow")

model_path = get_best_model_path()
model = mlflow.sklearn.load_model(model_path)
print(f" Loaded best model from: {model_path}")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int = Field(alias="InternetService_No")
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")
    PaymentMethod_Credit_card_automatic: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")
    TotalCharges_boxcox: float
    tenure_cbrt: float
    TotalServices: int
    HasPremiumSupport: int
    AvgMonthlySpend: float
@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        input_data = pd.DataFrame([data.model_dump(by_alias=True)])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        logging.info(f"INPUT: {data.model_dump(by_alias=True)} | PREDICTION: {prediction} | PROBABILITY: {probability:.4f}")

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_path": model_path,
            "message": "1 = Churn, 0 = No Churn"
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# === Entry Point ===
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
