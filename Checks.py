"""
# This file contains the code for preprocessing the data.
    A. Data Analysis:
        -> Dataset Analysis
        -> Univariate Analysis
        -> Bi-variate Analysis

    B. Data Cleaning: 
        -> Handling missing values
        -> Removing duplicates
        -> Outlier value Handling
"""
# Import Libraries
import csv                                              # Read and Write to CSV files
import pandas as pd                                     #
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns                                   
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')   

# Read both CSV Files
raw_data = pd.read_csv('Data/Original Data/raw_data.csv')
raw_data_copy = raw_data.copy()

validation_data = pd.read_csv('Data/Original Data/validation.csv')
validation_data_copy = validation_data.copy()


# ======================================================= A. DATA ANALYSIS PROCESSES ======================================================= #

# 3. Bi-variate Analysis
"""
    # When there are two variables in the data it is called bi-variate analysis. 
    # Data is analyzed to find the link between the two variables through causes and relationships.
    # Analyzing bi-variate data involves the following techniques:
        - Scatter plots
        - Correlation Coefficients
        - Covariance matrices
"""



# 5. For both dataset, perform outliers Data Handling


# 6. For both dataset, perform normalization of data distribution