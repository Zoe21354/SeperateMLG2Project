"""
# This file contains the code for preparing the data for analysis and model training.
    A. Data Analysis:
        -> Dataset Analysis
        -> Univariate Analysis
        -> Bi-variate Analysis

"""
# Import Libraries
import csv                                              # Read and Write to CSV files
import pandas as pd                                     # Manipulation and analysis of data
import numpy as np                                      # Mathematical operations
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns                                   
from sklearn.model_selection import train_test_split    # Splits the raw_data into two sets of data
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')   


# Read Unclean CSV Files
raw_data = pd.read_csv('Data/Original Data/raw_data.csv')
raw_data_copy = raw_data.copy()

validation_data = pd.read_csv('Data/Original Data/validation.csv')
validation_data_copy = validation_data.copy()


# ======================================================= A. DATA ANALYSIS PROCESSES ======================================================= #

# 1. Dataset Analysis
# Dataset Attributes:
print(f"Raw Data Columns: {raw_data_copy.columns}\n")
print(f"Validation Data Columns:{validation_data_copy.columns}\n")

"""
# Answer:
    - Raw Data Columns: 
        Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',      
        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'], dtype='object')

    - Validation Data Columns:
        Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',     
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'], dtype='object')

# Insights Gained:
    - The attribute names are inconsistent and will need standardizing in the data processing section.
    - Feature Variable (Independent variable): This variable stands alone and is not changed by other variables that are being measured. It is denoted as X in ML algorithms.
    - Target Variable (Dependent variable): This is the variable that is to be predicted. It is often denoted as Y in ML algorithms.
    - In both datasets there are 12 feature variables but only the raw_data dataset has 1 target variable.
    - The target variable in the raw_data dataset is the Loan_Status attribute.
    - This variable will be predicted using models for the validation_data dataset.
"""

# Dataset DataTypes:
print(f"Raw Dataset Datatypes:\n{raw_data_copy.dtypes}\n")
print(f"Validation Dataset Datatypes:\n{validation_data_copy.dtypes}\n")

"""
#Answers:
    - Raw Dataset Datatypes:
        Loan_ID               object
        Gender                object
        Married               object
        Dependents           float64
        Education             object
        Self_Employed         object
        ApplicantIncome        int64
        CoapplicantIncome      int64
        LoanAmount           float64
        Loan_Amount_Term     float64
        Credit_History       float64
        Property_Area         object
        Loan_Status           object
        dtype: object

    - Validation Dataset Datatypes:
        Loan_ID               object
        Gender                object
        Married               object
        Dependents            object
        Education             object
        Self_Employed         object
        ApplicantIncome        int64
        CoapplicantIncome      int64
        LoanAmount           float64
        Loan_Amount_Term     float64
        Credit_History       float64
        Property_Area         object
        dtype: object

# Insights Gained:
    - There is a discrepancy between the two datasets: the "Dependents" attribute is of datatype float64 in the raw_data.csv file but of datatype object in the validation_data.csv file. 
    - This could lead to potentially issues when modeling, as the model might be expecting the same data type for a given attribute.
    - This discrepancy will need to be fixed in the data processing section.
"""

# Dataset Shape:
print(f"Raw Data Shape:\n{raw_data_copy.shape}")
print(f"Validation Data Shape:\n{validation_data_copy.shape}")

"""
# Answers:
    - Raw Data Shape: (614, 13)
    - Validation Data Shape: (367, 12)

# Insights Gained:
    - Raw Data Shape: 614 rows and 13 columns
    - Validation Data Shape: 367 rows and 12 columns
"""

# 2. Univariate Analysis
"""
    # When there is just one variable in the data it is called univariate analysis. 
    # This is the most basic type of data analysis and finds patterns in the data.
    # Analyzing univariate data involves examining:
        - Frequency of data
        - Mean, mode, median, and range
"""
# Frequency and Bar charts of each Independent variable and the Dependant variable:
    #Get the count for each category in the variable
    #Normalize the data to get the proportion of the different categories in the variable (each count is divided by the total number of values)
    #Plot a bar chart to visually display the data


#Dependent Variable
count = raw_data_copy['Loan_Status'].value_counts(normalize = True)
chart = count.plot.bar(title = 'Loan_Status', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.69 or 69% of the people were approved for a loan (i.e Loan_Status = Yes)
    - 0.31 or 31% of the people were not approved for a loan (i.e Loan_Status = No)
"""

#Independent Variable (Categorical)
count = raw_data_copy['Gender'].value_counts(normalize = True)
chart = count.plot.bar(title = 'Gender', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.81 or 81% of the people are male (i.e Gender = Male)
    - 0.19 or 19% of the people are female (i.e Gender = Female)
"""

count = raw_data_copy['Married'].value_counts(normalize = True)
chart = count.plot.bar(title='Married', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.65 or 65% of the people were Married (i.e Married = Yes)
    - 0.35 or 35% of the people were not Married (i.e Married = No)
"""

count = raw_data_copy['Self_Employed'].value_counts(normalize = True)
chart = count.plot.bar(title='Self_Employed', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.14 or 14% of the people are self-employed (i.e Self_Employed = Yes)
    - 0.86 or 86% of the people are not self-employed (i.e Married = No)
"""

count = raw_data_copy['Credit_History'].value_counts(normalize = True)
chart = count.plot.bar(title='Credit_History', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.84 or 84% of the people have a credit history (i.e Credit_History = 1)
    - 0.16 or 16% of the people don't have a credit history (i.e Credit_History = 0)
"""

#Independent Variable (Ordinal)
count = raw_data_copy['Dependents'].value_counts('normalize = True')
chart = count.plot.bar(title='Dependents', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.58 or 58% of the people don't have Dependent (i.e Dependents = 0)
    - 0.17 or 17% of the people has only one Dependent (i.e Dependents = 1)
    - 0.17 or 17% of the people has two Dependents (i.e Dependents = 2)
    - 0.09 or 9% of the people has three or more Dependents (i.e Dependents = 3+)
"""

count =raw_data_copy['Education'].value_counts('normalize = True')
chart = count.plot.bar(title='Education', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.78 or 78% of the people have graduated (i.e Education = Graduate)
    - 0.22 or 22% of the people have not graduated (i.e Education = Not Graduate)
"""

count = raw_data_copy['Property_Area'].value_counts('normalize=True')
chart = count.plot.bar(title='Property_Area', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 0.38 or 38% of the people are located in the semi-urban area (i.e Property_Area = Semiurban)
    - 0.33 or 33% of the people are located in the urban area (i.e Property_Area = Urban)
    - 0.29 or 29% of the people are located in the rural area(i.e Property_Area = Rural)
"""

#Independent Variable (Nominal)
    # Use a distribution chart to visually see the distribution of the values in the attributes ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
plt.figure(1)
plt.subplot(121)
raw_data_copy.dropna() #Drop missing data in the attribute's data
sns.distplot(raw_data_copy['ApplicantIncome'])
plt.title('Distribution of Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['ApplicantIncome'].plot.box()
boxplot.set_title('Box Plot of Applicant Income')
boxplot.set_xlabel('Density')
plt.show()

plt.figure(2)
plt.subplot(121)
raw_data_copy.dropna() 
sns.distplot(raw_data_copy['CoapplicantIncome'])
plt.title('Distribution of Coapplicant Income')
plt.xlabel('Coapplicant Income')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['CoapplicantIncome'].plot.box()
boxplot.set_title('Box Plot of Coapplicant Income')
boxplot.set_xlabel('Density')
plt.show()


"""
#Insight Gained:
    - Both the distribution charts of the ApplicantIncome and CoapplicantIncome show a left-skewed distribution that indicates a majority of the applicants have lower incomes.
    - This pattern reflects income inequality within the applicant pool.
"""
plt.figure(3)
plt.subplot(121)
raw_data_copy.dropna() 
sns.distplot(raw_data_copy['LoanAmount'])
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['LoanAmount'].plot.box()
boxplot.set_title('Box Plot of Loan Amount')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - Overall the distribution of the data is fairly normal.
    - There are outliers in this attribute which could negatively impact the mean and distribution of the data 
    - These outliers will be treated in the data cleaning process
"""
plt.figure(4)
plt.subplot(121)
raw_data_copy.dropna() 
sns.distplot(raw_data_copy['Loan_Amount_Term'])
plt.title('Distribution of Loan Amount Term')
plt.xlabel('Loan Amount Term')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['Loan_Amount_Term'].plot.box()
boxplot.set_title('Box Plot of Loan Amount Term')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - The peak around 360 indicates a standard loan term.
    - Smaller peaks at lower values show that shorter loan terms are less common.
"""


# 3. Bi-variate Analysis
"""
    # When there are two variables in the data it is called bi-variate analysis. 
    # Data is analyzed to find the relationship between the dependent and independent variables.
    # Analyzing bi-variate data involves the following techniques:
        - Scatter plots and stacked bar graphs
        - Correlation Coefficients
        - Covariance matrices
    # The graphs created below will display how the Dependent Attribute ‘Loan_Status’ is distributed within each Independent Attribute, regardless of how many observations there are.
"""
# Categorical Independent Variables and Dependent Variable LoanAmount
# Loan_Status vs Gender
gender_table = pd.crosstab(raw_data_copy['Gender'], raw_data_copy['Loan_Status'])
gender_table.div(gender_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Gender Category')
plt.xlabel('Gender Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - The proportion of ‘Y’ loan status is slightly higher for males, indicating a marginally higher approval rate compared to females.
    - For both genders, the majority of the loan status is ‘Y’, suggesting that most applicants in the dataset were approved for a loan.
"""

# Loan_Status vs Married
married_table = pd.crosstab(raw_data_copy['Married'], raw_data_copy['Loan_Status'])
married_table.div(married_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Marriage Category')
plt.xlabel('Married Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - The ‘Yes’ category shows a higher proportion for the loan status ‘Yes’, suggesting that married individuals may have a better chance of loan approval.
    - Conversely, the ‘No’ category has a higher proportion for the loan status ‘No’, indicating that unmarried individuals may face more rejections.
"""

# Loan_Status vs Self_Employed
self_employed_table = pd.crosstab(raw_data_copy['Self_Employed'], raw_data_copy['Loan_Status'])
self_employed_table.div(self_employed_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Self Employment Category')
plt.xlabel('Self Employment Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - The ‘Yes’ loan status is present in both self-employment categories, but there is a slightly larger proportion of approvals for individuals who are not self-employed (‘No’) compared to those who are self-employed (‘Yes’).
    - Self-Employment Impact: The graph suggests that being self-employed might have a slight impact on loan approval rates, although the difference is not substantial.
"""

# Loan_Status vs Credit_History
credit_history_table = pd.crosstab(raw_data_copy['Credit_History'], raw_data_copy['Loan_Status'])
credit_history_table.div(credit_history_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Credit History Category')
plt.xlabel('Credit History Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - 
    - Individuals in Credit History Category ‘1’ have a higher proportion of getting approval for a loan, indicating a positive correlation between a good credit history and loan approval.
    - Category ‘0’ has a higher proportion of being rejected for a loan approval, suggesting that a poor credit history is associated with higher loan rejections.
"""



# Ordinal Independent Variables and Dependent Variable LoanAmount
# Loan_Status vs Dependents
dependents_table = pd.crosstab(raw_data_copy['Dependents'], raw_data_copy['Loan_Status'])
dependents_table.div(dependents_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Dependent Category')
plt.xlabel('Dependent Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - The ‘Yes’ loan status is present across all dependent categories, but there is a trend where the proportion of approvals decreases as the number of dependents increases.
    - The graph suggests that having more dependents might negatively impact the rate of loan approval.
"""

# Loan_Status vs Education
education_table = pd.crosstab(raw_data_copy['Education'], raw_data_copy['Loan_Status'])
education_table.div(education_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Education Category')
plt.xlabel('Education Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - A larger proportion of graduates have their loans approved (‘Y’) compared to non-graduates who have a higher proportion of being rejected (‘N’).
"""

# Loan_Status vs Property Area
property_area_table = pd.crosstab(raw_data_copy['Property_Area'], raw_data_copy['Loan_Status'])
property_area_table.div(property_area_table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Loan Status by Property Area Category')
plt.xlabel('Property Area Categories')
plt.ylabel('Loan Status')
plt.show()

"""
# Insight Gained:
    - 
    - The Semiurban areas have the highest proportion of approved loans (‘Y’), suggesting a favorable outcome for loan applicants in these areas.
    - The Rural area has the lowest proportion of approved loans, indicating potential challenges or stricter criteria for loan approval.
    - Urban Observations: The Urban area has a moderate proportion of approved loans, falling between the Rural and Semiurban areas.
"""

# Numerical Independent Variables and Dependent Variable LoanAmount
"""     
    #   The purpose of this section is to provide insight into how the income levels (both individually and combined with co-applicants) 
        relate to the likelihood of a loan being approved.
"""

# In order to determine the impact of the income on the Loan_Status, the mean income is calculated to determine who's loans were approved vs who's were not.
# Loan_Status vs Applicant_Income
raw_data_copy.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.title('Average Applicant Income by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Average Applicant Income')
plt.show()


# ApplicantIncome is categorized in loans within each income bracket.
# This will access whether different income levels, when the applicant income and the co-applicant income are added together, will influence the Loan approval rate.
# Binning will transform the continuous numerical variables into discrete categorical ‘bins’. 
# Income brackets such as "Low", "Average", "Above Average", and "High" are used to provide a qualitative understanding of the ranges in the data.

# Loan_Status vs Total Income
# Combine the applicant income and co-applicant income together
raw_data_copy['Total_Income']=raw_data_copy['ApplicantIncome']+raw_data_copy['CoapplicantIncome']

# Calculate the bin values
low = raw_data_copy['ApplicantIncome'].quantile(0.25) # 25th percentile
average = raw_data_copy['ApplicantIncome'].quantile(0.50) # 50th percentile
above_average = raw_data_copy['ApplicantIncome'].quantile(0.75) # 75th percentile
high = 81000

bins = [0, low, average, above_average, high]
group=['Low','Average','Above Average','High']

raw_data_copy['Total_Income_bin']=pd.cut(raw_data_copy['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(raw_data_copy['Total_Income_bin'],raw_data_copy['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Percentage of Total Income Per Income Bracket')
plt.xlabel('Total Income')
plt.ylabel('Percentage')
plt.show()


"""
# Insight Gained:
    - Low Income Approval: The proportion of loans approved for applicants with low total income is significantly lower than for other income groups.
    - Higher Income Approval: Applicants with average, high, and very high total income have a greater proportion of loan approvals.
    - Income Impact: The chart suggests that total income level may impact the likelihood of loan approval.
    - This analysis indicates that higher income levels are associated with better chances of loan approval, highlighting the importance of income in the loan decision process.
"""

# Loan amount is categorized in loans within each loan bracket.
# This will access whether different loan amounts will influence the Loan approval rate.
# Loan brackets such as "Low", "Average", and "High" are used to provide a qualitative understanding of the ranges in the data.

# Loan_Status vs Loan Amount
# Calculate the bin values
low = raw_data_copy['LoanAmount'].quantile(0.333) # 33.3th percentile
average = raw_data_copy['LoanAmount'].quantile(0.666) # 66.6th percentile
high = 700

bins = [0, low, average, high]
group=['Low','Average','High']

raw_data_copy['Loan_Amount_bin']=pd.cut(raw_data_copy['LoanAmount'],bins,labels=group)
Total_Income_bin=pd.crosstab(raw_data_copy['Loan_Amount_bin'],raw_data_copy['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Percentage of Loan Amount Per Loan Bracket')
plt.xlabel('Loan Amount')
plt.ylabel('Percentage')
plt.show()


"""
# Insight Gained:
    - Low and Average Loan Amounts: The proportion of approved loans is higher for these categories, indicating a greater likelihood of approval for smaller loan amounts.
    - High Loan Amount: The proportion of approved loans is lower for this category, suggesting that larger loan amounts may have a reduced chance of approval.
    - Therefore it can be said that loans with lower amounts are more likely to be approved.
"""

# Drop all bins created:
raw_data_copy=raw_data_copy.drop(['Loan_Amount_bin','Total_Income_bin','Total_Income'],axis=1)


# Using a Heatmap, the numerical attributes in the dataset is viewed to gain insight into the overall comparison through the colour shade variations
numeric_cols = validation_data_copy.select_dtypes(include=[np.number])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='PuRd')
plt.title('Correlation Heatmap')
plt.show()

"""
# Insight Gained:
    - Moderate Correlation: 
        ApplicantIncome and LoanAmount have a moderate positive correlation, suggesting that as applicant income increases, the loan amount tends to increase as well.

    - Credit History Impact: 
        The moderate positive correlation between Credit_History and Loan_Status indicates that applicants with a good credit history are more likely to have their loans approved.

    - Coapplicant Contribution: 
        While there is a positive correlation between LoanAmount and CoapplicantIncome, it is relatively weak, implying that co-applicant income has a lesser impact on the loan 
        amount compared to the primary applicant’s income.

    - Overall, the heatmap suggests that both income and credit history play significant roles in loan amount determination and approval. 
        The weaker correlation for CoapplicantIncome may indicate that lenders prioritize the primary applicant’s financial status.
"""