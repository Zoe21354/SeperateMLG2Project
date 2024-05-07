"""
# This file contains the code for training the models. 
    # This might include:
        1. defining the model architecture, 
        2. compiling the model,
        3. training the model
"""
import pandas as pd
from sklearn.model_selection import train_test_split    # Splits the raw_data into two sets of data
from sklearn.linear_model import LogisticRegression     # 
from sklearn.metrics import accuracy_score
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


# ================================================== A. SPLITTING THE RAW DATA INFORMATION ================================================== #
"""     
    # Dummy data is used to convert the categorical data into 0's and 1's  to make it easy to be quantified and compared in the future models
        -> Example: Gender has Male and Female categories
        -> Using 'dummies' from pandas, it converts them into Gender_Male = 1 and Gender_Female = 0
    # Training data set has weight 80% 0r 0.8
    # Testing data set has weight 20% or 0.2
    # 'random_state=42' is used for reproducibility, meaning if the code is run multiple times the same train/test split will occur every time.
"""

# Read Cleaned CSV Files
cleaned_raw_data = pd.read_csv('Data/Cleaned Data/cleaned_raw_data.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Define the independent variables (features) and the target variable
X = cleaned_raw_data_copy.drop('Loan_Status', axis=1)  # all columns except 'Loan_Status'
y = cleaned_raw_data_copy['Loan_Status']  # only 'Loan_Status' column

# Convert categorical variable in the X dataset(all columns except 'Loan_Status') into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" # Create new DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1) """

""" # Save the training and testing sets to CSV files
train_data.to_csv('Data/Split Data/train_data.csv', index=False)
test_data.to_csv('Data/Split Data/test_data.csv', index=False) """


# ================================================== B. TRAIN MODEL 1 ================================================== #
# Create a Logistic Regression model and Fit the model with the training data
model1 = LogisticRegression()
model1.fit(X_train, y_train)

# Calculate the accuracy score for the predictions of the model
test_predictions = model1.predict(X_test)
print(f"Accuracy Score for Predictions: {accuracy_score(y_test,test_predictions)}")

""" 
# Answer:
    Accuracy Score for Predictions: 0.7723577235772358

# Insight Gained:
    - The model shows it can accurately predict 77% of the Loan_Status values correctly.
"""

# ==============================================B. PREDICTIONS FOR MODEL 1 TEST DATA ============================================= #
# Read prediction CSV Files
predictions = pd.read_csv('Artifacts/Predictions.csv')