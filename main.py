""""""

import joblib
import warnings
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from dice_ml import Model
from dice_ml import Dice
from dice_ml import Data
from dice_ml.utils import helpers
import json
warnings.filterwarnings("ignore")

app = FastAPI()

# Set the templates directory
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def result(
    request: Request,
    Age: int = Form(...),
    AnnualIncome: int = Form(...),
    CreditScore: int = Form(...),
    EmploymentStatus: int = Form(...),
    EducationLevel: int = Form(...),
    Experience: int = Form(...),
    LoanAmount: int = Form(...),
    LoanDuration: int = Form(...),
    NumberOfDependents: int = Form(...),
    MonthlyDebtPayments: int = Form(...),
    NumberOfOpenCreditLines: int = Form(...),
    PaymentHistory: int = Form(...),
    LengthOfCreditHistory: int = Form(...),
    SavingsAccountBalance: int = Form(...),
    CheckingAccountBalance: int = Form(...),
    TotalAssets: int = Form(...),
    TotalLiabilities: int = Form(...),
    JobTenure: int = Form(...),
    NetWorth: int = Form(...)
):
    # Load MLP model
    mlp = joblib.load('model/loan_approval_model.pkl')

    # Load scaler
    scaler = joblib.load('model/scaled.pkl')

    # Load data 
    X = joblib.load('model/data.pkl')

    # Load y
    y = joblib.load('model/labels.pkl')

    # Load training data
    X_train = joblib.load('model/train_data.pkl')

    input_data = pd.DataFrame({
        'Age': [Age],
        'AnnualIncome': [AnnualIncome],
        'CreditScore': [CreditScore],
        'EmploymentStatus': [EmploymentStatus],
        'EducationLevel': [EducationLevel],
        'Experience': [Experience],
        'LoanAmount': [LoanAmount],
        'LoanDuration': [LoanDuration],
        'NumberOfDependents': [NumberOfDependents],
        'MonthlyDebtPayments': [MonthlyDebtPayments],
        'NumberOfOpenCreditLines': [NumberOfOpenCreditLines],
        'PaymentHistory': [PaymentHistory],
        'LengthOfCreditHistory': [LengthOfCreditHistory],
        'SavingsAccountBalance': [SavingsAccountBalance],
        'CheckingAccountBalance': [CheckingAccountBalance],
        'TotalAssets': [TotalAssets],
        'TotalLiabilities': [TotalLiabilities],
        'JobTenure': [JobTenure],
        'NetWorth': [NetWorth]
    })

    # Apply the same StandardScaler used during training
    input_data_sacled = scaler.transform(input_data)
    
    # Return Status of application
    loan_status = mlp.predict(input_data_sacled)
    print(f"Loan Status: {loan_status}")

    # Create Explianer object
    explainer = LimeTabularExplainer(
        training_data=X_train,
        training_labels=y,
        feature_names=X.columns,
        class_names=['Approved', 'Not Approved'],
        discretize_continuous=True
    )

    # Explain a local prediction
    row = scaler.inverse_transform(input_data_sacled[0].reshape(1, -1))
    explain = explainer.explain_instance(row[0], mlp.predict_proba, num_features=5)

    # Display explantation as a HTML file
    with open('static/explainer.html', 'w') as f:
        f.write(explain.as_html())

    # Transorm dataset to DiCE Data
    X['LoanApproval'] = y
    continuous_features = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration', 
                        'MonthlyDebtPayments', 'NumberOfOpenCreditLines', 'PaymentHistory', 
                        'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance', 
                        'TotalAssets', 'TotalLiabilities', 'JobTenure', 'NetWorth']
    dice_data = Data(dataframe=X, continuous_features=continuous_features, outcome_name='LoanApproval')

    # Create DiCE model
    dice_model = Model(model=mlp, backend='sklearn')

    # Create dice explainer
    dice_explain = Dice(dice_data, dice_model, method='random')

    dice_counterfactual = dice_explain.generate_counterfactuals(input_data, total_CFs=5, desired_class='opposite', verbose=False)
    output = json.loads(dice_counterfactual.to_json())

    return templates.TemplateResponse("result.html", {
        "request": request, 
        "test_data": output["test_data"][0][0],
        "feature_names": output["feature_names"],
        "cfs_list": output["cfs_list"]
        })