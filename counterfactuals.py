"""Create counterfactuals that explain a model."""


import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load MLP model
mlp = joblib.load('loan_approval_model.pkl')

# Load scaler
scaler = joblib.load('scaled.pkl')

# Test One Input: 
test_one = pd.DataFrame({
    'Age': 30,
    'AnnualIncome': [80000],
    'CreditScore': [750],
    'EmploymentStatus': [0],
    'EducationLevel': [2],
    'Experience': [8],
    'LoanAmount': [15000],
    'LoanDuration': [60],
    'NumberOfDependents': [2],
    'MonthlyDebtPayments': [1200],
    'NumberOfOpenCreditLines': [3],
    'PaymentHistory': [10],
    'LengthOfCreditHistory': [12],
    'SavingsAccountBalance': [5000],
    'CheckingAccountBalance': [2000],
    'TotalAssets': [100000],
    'TotalLiabilities': [30000],
    'JobTenure': [5],
    'NetWorth': [70000]
})

# Apply the same StandardScaler used during training
test_one_scaled = scaler.transform(test_one)

# Predict with the trained model
# 1: Approved, 0: Not Approved
prediction = mlp.predict(test_one_scaled)
print(f"Prediction: {prediction}")
