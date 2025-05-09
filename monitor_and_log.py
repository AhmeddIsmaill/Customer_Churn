
import datetime
import pandas as pd
import os

def log_prediction(input_data: dict, prediction: int, probability: float):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction": prediction,
        "probability": round(probability, 4),
        **input_data
    }
    
    log_path = "user_logs.csv"
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

if __name__ == "__main__":
    sample_input = {
        'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0, 'PhoneService': 1,
        'PaperlessBilling': 1, 'MonthlyCharges': 75.20, 'InternetService_Fiber optic': 1,
        'InternetService_No': 0, 'Contract_One year': 0, 'Contract_Two year': 1,
        'PaymentMethod_Credit card (automatic)': 1, 'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0, 'TotalCharges_boxcox': 4.21, 'tenure_cbrt': 2.46,
        'TotalServices': 6, 'HasPremiumSupport': 1, 'AvgMonthlySpend': 74.15
    }
    log_prediction(sample_input, prediction=0, probability=0.135)
