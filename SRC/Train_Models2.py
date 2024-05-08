# ================================================== A. FEATURE ENGINEERING ================================================== #
# Completed in Feature_Engineering.py
# 1. Creating new features based on domain knowledge or insights from the data
# 2. Removing irrelevant or redundant features
# 3. Handling missing values or outliers in the data
# 4. Scaling or normalizing numerical features
# 5. Encoding categorical features appropriately

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# Read Cleaned CSV Files
cleaned_raw_data = pd.read_csv('Data/Cleaned Data/cleaned_raw_data.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Define the independent variables (features) and the target variable
X = cleaned_raw_data_copy.drop('Loan_Status', axis=1)  
y = cleaned_raw_data_copy['Loan_Status']  

# Convert categorical variable in the X dataset into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================================== B. TRAIN MODEL 2 ================================================== #
# Define the deep learning model architecture
model2 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model2.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
loss, accuracy = model2.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save Model 2 as a pickle file
with open('Artifacts/model2.pkl', 'wb') as f:
    pickle.dump(model2, f)

# Create dummy feature importance values
feature_importance_train = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.random.rand(X_train.shape[1])})
feature_importance_test = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.random.rand(X_test.shape[1])})

# Save feature importance values to CSV files
feature_importance_train.to_csv('Artifacts/feature_importance_train.csv', index=False)
feature_importance_test.to_csv('Artifacts/feature_importance_test.csv', index=False)