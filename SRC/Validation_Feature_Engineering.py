import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore') 


# Read Cleaned CSV Files
Validation_data = pd.read_csv('Data/Cleaned Data/cleaned_validation_data.csv')
Validation_data_copy = Validation_data.copy()

# Convert categorical variable in the X dataset(all columns except 'Loan_Status') into dummy variables
Validation_data_copy = pd.get_dummies(Validation_data_copy)

# Feature 1: Total Income
Validation_data_copy['Total_Income']=Validation_data_copy['Applicant_Income']+Validation_data_copy['Coapplicant_Income']

#Distribution normalization
Validation_data_copy['Total_Income_Log']=np.log(Validation_data_copy['Total_Income'])


# Feature 2: Equated Monthly Installment (EMI)
Validation_data_copy['EMI']=Validation_data_copy['Loan_Amount']/Validation_data_copy['Loan_Amount_Term']

# Feature 3: Income_After_EMI
Validation_data_copy['Income_After_EMI']=Validation_data_copy['Total_Income']-(Validation_data_copy['EMI']*1000)


# Remove all features that created the new features
Validation_data_copy=Validation_data_copy.drop(['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term'],axis=1)

print(f"Validation Data Columns: {Validation_data_copy.columns}\n")
"""
Validation Data Columns: 
    Index(['Credit_History', 'Loan_Amount_Log', 'Gender_Female', 'Gender_Male',
    'Married_No', 'Married_Yes', 'Dependents_0', 'Dependents_1',
    'Dependents_2', 'Dependents_3+', 'Education_Graduate',
    'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
    'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban',
    'Total_Income', 'Total_Income_Log', 'EMI', 'Income_After_EMI'],
    dtype='object')
"""

# Store new Features in CSV files
Validation_data_copy.to_csv('Artifacts/Feature_Importance_validation_data.csv', index=False)