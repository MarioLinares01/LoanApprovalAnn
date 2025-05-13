# LoanApprovalAnn
A loan approval system that leverages a MLP Classifier, LIME, and Counterfactuals

## Live Demo
https://loanapprovalann.onrender.com

## Installation for local deployment

1) Create a virtual environment 
```
python3 -m venv venv
```

2) Activate the virtual environment
```
source venv/bin/activate
```

3) Install requirements
```
pip3 install -r requirements.txt
```

4) Run the application
```
uvicorn main:app
```

5) Visit http://127.0.0.1:8000


#### Disclaimer 
This project is intended for educational and demonstration purposes only. It is not designed, tested, or certified for use in real-world financial decision-making. Any predictions, recommendations, or explanations provided by the system should not be interpreted as professional financial advice or used to inform actual loan or credit decisions. The underlying models and data may not reflect real-world conditions, regulatory requirements, or ethical standards mandated in the financial industry. Users should consult qualified financial professionals or institutions for actual loan-related matters.
