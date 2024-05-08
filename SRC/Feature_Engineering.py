import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns  
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


train_data= pd.read_csv('Data/Split Data/train_data.csv')
test_data=pd.read_csv('Data/Split Data/test_data.csv')


# =============================================== FEATURE ENGINEERING MODEL 1 =============================================== #
"""     
    # Feature engineering transforms or combines raw data into a format that can be easily understood by machine learning models.
    # Creates predictive model features, also known as a dimensions or variables, to generate model predictions.
    # This highlights the most important patterns and relationships in the data, which then assists the machine learning model to learn from the data more effectively.
"""

# Feature 1: Total Income
train_data['Total_Income']=train_data['Applicant_Income']+train_data['Coapplicant_Income']
test_data['Total_Income']=test_data['Applicant_Income']+test_data['Coapplicant_Income']

#Distribution normalization
    # Decreases the affects of extreme values
sns.distplot(train_data['Total_Income'])
plt.title('Distribution of Total Income')
plt.xlabel('Total Income')
plt.ylabel('Density')
plt.show()

train_data['Total_Income_Log']=np.log(train_data['Total_Income'])
test_data['Total_Income_Log']=np.log(test_data['Total_Income'])

sns.distplot(train_data['Total_Income_Log'])
plt.title('Distribution of Total Income Log')
plt.xlabel('Total Income Log')
plt.ylabel('Density')
plt.show()


# Feature 2: Equated Monthly Installment (EMI)
    # The feature ‘EMI’ is created by dividing the ‘LoanAmount’ by the ‘Loan_Amount_Term’. 
    # This is to get the monthly payment amount for a loan, given the total loan amount and the term of the loan. 
    # This will give an indication of the individuals monthly financial obligation towards the loan.

train_data['EMI']=train_data['Loan_Amount']/train_data['Loan_Amount_Term']
test_data['EMI'] = test_data['Loan_Amount']/test_data['Loan_Amount_Term']

sns.distplot(train_data['EMI'])
plt.title('Distribution of Equated Monthly Installments')
plt.xlabel('Equated Monthly Installment')
plt.ylabel('Density')
plt.show()


# Feature 3: Balanced Income
    # The feature "Balanced Income" is created by dividing the ‘LoanAmount’ by the ‘Loan_Amount_Term’. 
    # This is to get the monthly payment amount for a loan, given the total loan amount and the term of the loan. 
    # This will give an indication of the individuals monthly financial obligation towards the loan.
    # 'EMI' is multiplied with 1000 to make the unit equal to 'Total_Income'.

train_data['Income_After_EMI']=train_data['Total_Income']-(train_data['EMI']*1000)
test_data['Income_After_EMI']=test_data['Total_Income']-(test_data['EMI']*1000)

sns.distplot(train_data['Income_After_EMI'])
plt.title('Distribution of Income After EMI')
plt.xlabel('Income After EMI')
plt.ylabel('Density')
plt.show()

# Remove all features that created the new features
    # The correlation between those old feature and the new features are very high.
    # Logistic regression assume that the variables are not highly correlated.
    # Due to this the excess noise in the datasets are removed.

train_data=train_data.drop(['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term'],axis=1)
test_data=test_data.drop(['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term'],axis=1)

print(f"Training Data Columns: {train_data.columns}\n")
print(f"Testing Data Columns:{test_data.columns}\n")

"""
Training Data Columns: 
    Index(['Dependents', 'Credit_History', 'Loan_Amount_Log', 'Gender_Female','Gender_Male', 'Married_No', 'Married_Yes', 'Education_Graduate',
    'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes','Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban',
    'Loan_Status', 'Total_Income', 'Total_Income_Log', 'EMI','Income_After_EMI'],dtype='object')

Testing Data  Columns:Index(['Dependents', 'Credit_History', 'Loan_Amount_Log', 'Gender_Female','Gender_Male', 'Married_No', 
    'Married_Yes', 'Education_Graduate','Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes','Property_Area_Rural', 
    'Property_Area_Semiurban', 'Property_Area_Urban','Loan_Status', 'Total_Income', 'Total_Income_Log', 'EMI', 'Income_After_EMI'], dtype='object')
"""

# Store new Features in CSV files
train_data.to_csv('Artifacts/Feature_Importance_train_data_NF_Model1.csv', index=False)
test_data.to_csv('Artifacts/Feature_Importance_test_data_NF_Model1.csv', index=False)


# =============================================== FEATURE ENGINEERING MODEL 2 =============================================== #
# Feature 1: Total Income
train_data['Total_Income'] = train_data['Applicant_Income'] + train_data['Coapplicant_Income']
test_data['Total_Income'] = test_data['Applicant_Income'] + test_data['Coapplicant_Income']

# Distribution normalization
sns.distplot(train_data['Total_Income'])
plt.title('Distribution of Total Income')
plt.xlabel('Total Income')
plt.ylabel('Density')
plt.show()

train_data['Total_Income_Log'] = np.log(train_data['Total_Income'])
test_data['Total_Income_Log'] = np.log(test_data['Total_Income'])

sns.distplot(train_data['Total_Income_Log'])
plt.title('Distribution of Total Income Log')
plt.xlabel('Total Income Log')
plt.ylabel('Density')
plt.show()

# Feature 2: Equated Monthly Installment (EMI)
train_data['EMI'] = train_data['Loan_Amount'] / train_data['Loan_Amount_Term']
test_data['EMI'] = test_data['Loan_Amount'] / test_data['Loan_Amount_Term']

sns.distplot(train_data['EMI'])
plt.title('Distribution of Equated Monthly Installments')
plt.xlabel('Equated Monthly Installment')
plt.ylabel('Density')
plt.show()

# Feature 3: Balanced Income
train_data['Income_After_EMI'] = train_data['Total_Income'] - (train_data['EMI'] * 1000)
test_data['Income_After_EMI'] = test_data['Total_Income'] - (test_data['EMI'] * 1000)

sns.distplot(train_data['Income_After_EMI'])
plt.title('Distribution of Income After EMI')
plt.xlabel('Income After EMI')
plt.ylabel('Density')
plt.show()

# Remove all features that created the new features
train_data = train_data.drop(['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term'], axis=1)
test_data = test_data.drop(['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term'], axis=1)

print(f"Training Data Columns: {train_data.columns}\n")
print(f"Testing Data Columns: {test_data.columns}\n")

# Store new Features in CSV files
train_data.to_csv('Artifacts/Feature_Importance_train_data_NF_Model2.csv', index=False)
test_data.to_csv('Artifacts/Feature_Importance_test_data_NF_Model2.csv', index=False)