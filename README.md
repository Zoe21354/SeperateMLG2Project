ReadMe.md: This file provides:
    1. An overview of the project, 
    2. The problem statement, 
    3. The approach taken,
    4. The file structure
    5. Instructions on how to run the code.

# Machine Learning 381 - GroupF - Streamlining BC Finance’s Home Loan Eligibility Process

BC Finance Company provides financial services across all home loan categories. Offering services to clients in urban, semi-urban, and rural areas, the organization has many facets. The organization currently uses an ineffective manual procedure to validate customers' eligibility. The procedure entails the client submitting an application for a home loan by answering questions and supplying personal information. These responses must then go through a lengthy validation process and this can be a problem for handling multiple applications leading to decreased customer satisfaction, manual errors and lengthy application times which could lead to customers seeking other financial institutes to provide financial services for their needs.

The organization is working to create an automated system that can accurately determine a customer's eligibility for a home loan in real time in order to address this problem. To ascertain if a customer is eligible for a loan, this system will examine a number of customer variables, including gender, marital status, education, number of dependents, income, loan amount, credit history, and others.

The principal aim is to divide clients into discrete categories according to their loan quantum eligibility. By doing this, BC Finance hopes to more successfully target these consumer segments and provide them with loan products and services that are customized to their unique requirements and preferences. BC Finance hopes to improve client happiness, reduce manual errors, and streamline its lending procedures for long-term profitability and growth by putting in place an automated loan qualifying system.

## **Hypothesis:**
The aim of this project is to use machine learning to transform BC Finance's loan approval process. BC Finance seeks to mitigate the inefficiencies linked to manual validation, including longer application periods, higher error rates, and lower customer satisfaction, by automating the real-time eligibility evaluation process. By using this automated approach, BC Finance hopes to improve resource allocation, boost operational efficiency, and ultimately become more competitive in the financial market.

The prepare_data.py file is essential in this situation as it organizes several data pretreatment and exploratory analysis activities. By carefully going over the dataset, which includes factors like gender, marital status, income levels, credit history, and property location, this script reveals important information that serves as the foundation for the phases of model construction and hypothesis formulation that follow.

Hypothesis 1
Hypothesis: The likelihood of loan approval is positively impacted by having a good credit history. 
Justification: Bi-variate analysis shows a moderately positive correlation between Credit_History and Loan_Status, indicating
that applicants with a good credit history are more likely to have their loans approved. 

Hypothesis 2
Hypothesis: Loan amounts in the low to average range are more likely to be approved than high loan amounts. 
Justification: Bi-variate analysis shows that the proportion of approved loans is higher for low and average loan amounts, indicating a greater likelihood of approval for smaller loan amounts. 

Hypothesis 3
Hypothesis: An applicant's marital status may have an impact on loan approval rates. 
Justification: Bi-variate analysis displays differences in loan approval rates for married individuals compared to unmarried individuals.

Hypothesis 4
Hypothesis: The type of property—rural, semi-urban, or urban—may affect the likelihood of a loan being approved.
Justification: variable property areas have variable loan approval rates, as shown by univariate analysis. As an illustration, the percentage of loans that are authorized is higher in semi-urban areas than in urban and rural areas.

Hypothesis 5
Hypothesis: There is a positive correlation between income levels and loan acceptance rates.
Justification: According to univariate research, applicants with higher earnings typically receive a higher percentage of loan approvals. This implies that judgments about loan approval may be significantly influenced by an individual's income level.