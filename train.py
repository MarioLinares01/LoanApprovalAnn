"""The training of a scikit-learn MLP that predicts a loan decision."""


import pandas as pd
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# Read dataset as a DataFrame
df = pd.read_csv('dataset/loan_data.csv')

# Drop rows with missing values or unnecessary features
df = df.dropna()

# features and target
X = df.drop(columns=['LoanApproved'])
y = df['LoanApproved']

# Convert EmploymentStatus to numerical
# EmploymentStatus => (employed 0, self-employed 1, unemployed )
X['EmploymentStatus'] = X['EmploymentStatus'].replace({
    'Employed': 0,
    'Self-Employed': 1,
    'Unemployed': 2
})

# Convert EducationLevel to numerical 
# High School 0, Associate 1, Bachelor 2, Master 3, Doctorate 4
X['EducationLevel'] = X['EducationLevel'].replace({
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
})

# Convert MaritalStatus to numerical data type
X['MaritalStatus'] = X['MaritalStatus'].replace({
    'Married': 0,
    'Single': 1,
    'Divorced': 2,
    'Widowed': 3
})

# Convert HomeOwnershipStatus to numerical data type
X['HomeOwnershipStatus'] = X['HomeOwnershipStatus'].replace({
    'Mortgage': 0,
    'Rent': 1,
    'Own': 2,
    'Other': 3
})

# Convert LoanPurpose to numerical data type
X['LoanPurpose'] = X['LoanPurpose'].replace({
    'Home': 0,
    'Debt Consolidation': 1,
    'Auto': 2,
    'Education': 3,
    'Other': 4
})

# Drop some columns
X = X.drop(columns='MaritalStatus')
X = X.drop(columns='HomeOwnershipStatus')
X = X.drop(columns='NumberOfCreditInquiries')
X = X.drop(columns='LoanPurpose')
X = X.drop(columns='PreviousLoanDefaults')
X = X.drop(columns='BankruptcyHistory')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0005,
                    alpha=0.001,
                    max_iter=300,
                    batch_size=128,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42)

# Train model
mlp.fit(X_train, y_train)

# Predict on test set
y_pred = mlp.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}')

print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(mlp, 'loan_approval_model.pkl')
# Save scaled datat
joblib.dump(scaler, 'scaled.pkl')
