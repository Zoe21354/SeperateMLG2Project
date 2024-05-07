"""
# This file contains the code for preprocessing the data.
    A. Data Cleaning: 
        -> Handling missing values
        -> Removing duplicates
        -> Outlier value Handling
"""
# Import Libraries
import csv                                              # Read and Write to CSV files
import pandas as pd                                     # Manipulation and analysis of data
import numpy as np                                      # Mathematical operations
from scipy import stats                                 # Statistical functions
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns                                   
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')   

# Read both CSV Files
raw_data = pd.read_csv('Data/Original Data/raw_data.csv')
raw_data_copy = raw_data.copy()

validation_data = pd.read_csv('Data/Original Data/validation.csv')
validation_data_copy = validation_data.copy()


# ======================================================= A. DATA CLEANING PROCESSES ======================================================= #

# 1. Attribute Name Standardization Process for both dataset
#Create a dictionary to map the old column names to the new ones to format the column names:
raw_column_name_mapping = {
    'Loan_ID': 'Loan_ID',
    'Gender': 'Gender',
    'Married': 'Married',
    'Dependents': 'Dependents',
    'Education': 'Education',
    'Self_Employed': 'Self_Employed',
    'ApplicantIncome': 'Applicant_Income',
    'CoapplicantIncome': 'Coapplicant_Income',
    'LoanAmount': 'Loan_Amount',
    'Loan_Amount_Term': 'Loan_Amount_Term',
    'Credit_History': 'Credit_History',
    'Property_Area': 'Property_Area',
    'Loan_Status': 'Loan_Status'
}

validation_column_name_mapping = {
    'Loan_ID': 'Loan_ID',
    'Gender': 'Gender',
    'Married': 'Married',
    'Dependents': 'Dependents',
    'Education': 'Education',
    'Self_Employed': 'Self_Employed',
    'ApplicantIncome': 'Applicant_Income',
    'CoapplicantIncome': 'Coapplicant_Income',
    'LoanAmount': 'Loan_Amount',
    'Loan_Amount_Term': 'Loan_Amount_Term',
    'Credit_History': 'Credit_History',
    'Property_Area': 'Property_Area'
}

#Replace the original column names with the formatted names: 
raw_data_copy.rename(columns=raw_column_name_mapping, inplace=True)
validation_data_copy.rename(columns=validation_column_name_mapping, inplace=True)


# 2. For both dataset, check for missing values
print(f"Number of Missing Values in raw_data_copy: {raw_data_copy.isnull().sum()}\n")
print(f"Number of Missing Values in validation_data_copy: {raw_data_copy.isnull().sum()}")

""" Answers:  
Number of Missing Values in raw_data_copy:
Loan_ID                0
Gender                13
Married                3
Dependents            15
Education              0
Self_Employed         32
Applicant_Income       0
Coapplicant_Income     0
Loan_Amount           22
Loan_Amount_Term      14
Credit_History        50
Property_Area          0
Loan_Status            0
dtype: int64

Number of Missing Values in validation_data_copy:
Loan_ID                0
Gender                13
Married                3
Dependents            15
Education              0
Self_Employed         32
Applicant_Income       0
Coapplicant_Income     0
Loan_Amount           22
Loan_Amount_Term      14
Credit_History        50
Property_Area          0
dtype: int64"""


#Fill in missing values
""" 
    # Based on the missing value check above, it was noted that there are missing values in the following attributes for both data sets:
        - Gender
        - Married
        - Dependents
        - Self_Employed
        - LoanAmount
        - Loan_Amount_Term
        - Credit_History

    # In order to fill in the missing values the attributes need to be split into Categorical and Numerical Variables.
        # For Categorical Variables use the mode of all the values in the attribute to fill in the missing data values:
            - Gender (Male or Female)
            - Married (Yes or No)
            - Dependents (0, 1, 2, or 3+)
            - Self_Employed (Yes or No)
            - Credit_History (1 or 0)


        # For Numerical Variables use either the mean or median of the values in the attribute to fill in the missing data values
            - Loan_Amount
            - Loan_Amount_Term
"""
#=== CATEGORICAL VARIABLES === 
raw_data_copy['Gender'].fillna(raw_data_copy['Gender'].mode()[0],inplace=True)
raw_data_copy['Married'].fillna(raw_data_copy['Married'].mode()[0],inplace=True)
raw_data_copy['Dependents'].fillna(raw_data_copy['Dependents'].mode()[0],inplace=True)
raw_data_copy['Self_Employed'].fillna(raw_data_copy['Self_Employed'].mode()[0],inplace=True)
raw_data_copy['Credit_History'].fillna(raw_data_copy['Credit_History'].mode()[0],inplace=True)

validation_data_copy['Gender'].fillna(validation_data_copy['Gender'].mode()[0],inplace=True)
validation_data_copy['Married'].fillna(validation_data_copy['Married'].mode()[0],inplace=True)
validation_data_copy['Dependents'].fillna(validation_data_copy['Dependents'].mode()[0],inplace=True)
validation_data_copy['Self_Employed'].fillna(validation_data_copy['Self_Employed'].mode()[0],inplace=True)
validation_data_copy['Credit_History'].fillna(validation_data_copy['Credit_History'].mode()[0],inplace=True)


#=== NUMERICAL VARIABLES ===
# Median is used instead of mean due to the outliers in the attributes data which could negatively impact the outcome
raw_data_copy['Loan_Amount'].fillna(raw_data_copy['Loan_Amount'].median(),inplace=True)
raw_data_copy['Loan_Amount_Term'].fillna(raw_data_copy['Loan_Amount_Term'].median(),inplace=True)

validation_data_copy['Loan_Amount'].fillna(validation_data_copy['Loan_Amount'].median(),inplace=True)
validation_data_copy['Loan_Amount_Term'].fillna(validation_data_copy['Loan_Amount_Term'].median(),inplace=True)

#Check the file to see whether the missing values have been added
print(f"Number of Missing Values in raw_data_copy: {raw_data_copy.isnull().sum()}\n")
print(f"Number of Missing Values in validation_data_copy: {raw_data_copy.isnull().sum()}")

"""Answer: 
Number of Missing Values in raw_data_copy: 
Loan_ID               0
Gender                0
Married               0
Dependents            0
Education             0
Self_Employed         0
Applicant_Income      0
Coapplicant_Income    0
Loan_Amount           0
Loan_Amount_Term      0
Credit_History        0
Property_Area         0
Loan_Status           0
dtype: int64

Number of Missing Values in validation_data_copy: 
Loan_ID               0
Gender                0
Married               0
Dependents            0
Education             0
Self_Employed         0
Applicant_Income      0
Coapplicant_Income    0
Loan_Amount           0
Loan_Amount_Term      0
Credit_History        0
Property_Area         0
dtype: int64"""


# 3. For both dataset, check for duplicate records and remove them from the dataset
print(f"Number of duplicate rows in raw_data_copy: {raw_data_copy.duplicated().sum()}")
print(f"Number of duplicate rows in validation_data_copy: {validation_data_copy.duplicated().sum()}")

""" Answers:
Number of duplicate rows in raw_data_copy: 0
Number of duplicate rows in validation_data_copy: 0
"""

    # There are no duplicate records in either of the dataset, as a result no records need to be dropped 
        #raw_data_copy = raw_data_copy.drop_duplicates()
        #validation_data_copy = validation_data_copy.drop_duplicates() 


# 4. For both dataset, perform outliers Data Handling using log transformation
#Loan Amount
raw_data_copy['Loan_Amount_Log'] = np.log(raw_data_copy['Loan_Amount'])
validation_data_copy['Loan_Amount_Log'] = np.log(validation_data_copy['Loan_Amount'])

#Plot to verify the changes
plt.figure(1)
plt.subplot(121)
sns.distplot(raw_data_copy['Loan_Amount_Log'])
plt.title('Log Transformed Loan Amount')
plt.subplot(122)
sns.boxplot(raw_data_copy['Loan_Amount_Log'])
plt.title('Boxplot of Log Transformed Loan Amount')
plt.show()

#Plot to verify the changes
plt.figure(1)
plt.subplot(121)
sns.distplot(validation_data_copy['Loan_Amount_Log'])
plt.title('Log Transformed Loan Amount')
plt.subplot(122)
sns.boxplot(validation_data_copy['Loan_Amount_Log'])
plt.title('Boxplot of Log Transformed Loan Amount')
plt.show()


# 5. For both dataset, transformation Data
#The Dependent value '3+' is replaced by '3' as logistic regression models only tak numerical values.
raw_data_copy['Dependents'].replace('+3',3,inplace=True)
validation_data_copy['Dependents'].replace('+3',3,inplace=True)

#Convert the datatype of the attribute 'Dependents' in validation_data_copy to object
raw_data_copy['Dependents'] = raw_data_copy['Dependents'].astype('object')
print(f"Dependents datatype:\n{raw_data_copy['Dependents'].dtypes}\n")
"""Answer: Dependents datatype: object"""

#The Loan_Status values 'Yes and No' are replaced by '1 and 0' as logistic regression models only tak numerical values.
raw_data_copy['Loan_Status'].replace('N',0,inplace=True)
raw_data_copy['Loan_Status'].replace('Y',1,inplace=True)


# 6. For both dataset, drop the Loan_ID column as it is does not affect the Dependent variable Loan_Status
raw_data_copy.drop('Loan_ID',axis=1,inplace=True)
validation_data_copy.drop('Loan_ID',axis=1,inplace=True)


# 7. For both datasets, write the new datasets to CSV files
raw_data_copy.to_csv('Data/Cleaned Data/cleaned_raw_data.csv', index=False)
validation_data_copy.to_csv('Data/Cleaned Data/cleaned_validation_data.csv', index=False)