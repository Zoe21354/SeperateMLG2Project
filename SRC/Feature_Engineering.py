import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns  
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


train_data= pd.read_csv('Data/Split Data/train_data.csv')
test_data=pd.read_csv('Data/Split Data/test_data.csv')


# ================================================== FEATURE ENGINEERING ================================================== #
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