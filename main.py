""""""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Set the templates directory
templates = Jinja2Templates(directory="templates")

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
    input_data = {
        "Age": Age,
        "AnnualIncome": AnnualIncome,
        "CreditScore": CreditScore,
        "EmploymentStatus": EmploymentStatus,
        "EducationLevel": EducationLevel,
        "Experience": Experience,
        "LoanAmount": LoanAmount,
        "LoanDuration": LoanDuration,
        "NumberOfDependents": NumberOfDependents,
        "MonthlyDebtPayments": MonthlyDebtPayments,
        "NumberOfOpenCreditLines": NumberOfOpenCreditLines,
        "PaymentHistory": PaymentHistory,
        "LengthOfCreditHistory": LengthOfCreditHistory,
        "SavingsAccountBalance": SavingsAccountBalance,
        "CheckingAccountBalance": CheckingAccountBalance,
        "TotalAssets": TotalAssets,
        "TotalLiabilities": TotalLiabilities,
        "JobTenure": JobTenure,
        "NetWorth": NetWorth,
    }
    return templates.TemplateResponse("result.html", {
        "request": request,
        "input_data": input_data,
    })