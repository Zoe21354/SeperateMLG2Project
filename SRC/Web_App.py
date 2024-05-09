import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import pickle

# Initialise the Dash App
app = dash.Dash(__name__)

# Load the model
with open('Artifacts/Model_2.pkl', 'rb') as f:
    model = pickle.load(f)

# Define CSS styles
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css']
colors = {
    'background': '#f7f7f7',  # Light gray background
    'text': '#333333',  # Dark gray text
    'accent': '#6e40c9',  # Purple accent color
}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define App Layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '30px'}, children=[
    html.H1("Loan Eligibility Predictor", style={'textAlign': 'center', 'color': colors['accent']}),
    html.Div(children=[
        html.Label("Gender:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='gender',
            options=[
                {'label': 'Male', 'value': 1},
                {'label': 'Female', 'value': 0}
            ],
            value=1
        ),
        html.Label("Married:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='married',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1
        ),
        html.Label("Dependents:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='dependents',
            options=[
                {'label': '0', 'value': 0},
                {'label': '1', 'value': 1},
                {'label': '2', 'value': 2},
                {'label': '3+', 'value': 3}
            ],
            value=0
        ),
        html.Label("Education:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='education',
            options=[
                {'label': 'Graduate', 'value': 'Graduate'},
                {'label': 'Non Graduate', 'value': 'Non Graduate'}
            ],
            value='Graduate'
        ),
        html.Label("Self Employed:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='self_employed',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label("Applicant Income:", style={'color': colors['text']}),
        dcc.Input(id='applicant_income', type='number', value=0),
        html.Label("Coapplicant Income:", style={'color': colors['text']}),
        dcc.Input(id='coapplicant_income', type='number', value=0),
        html.Label("Loan Amount:", style={'color': colors['text']}),
        dcc.Input(id='loan_amount', type='number', value=0),
        html.Label("Loan Amount Term:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='loan_amount_term',
            options=[
                {'label': '12 months', 'value': 12},
                {'label': '36 months', 'value': 36},
                {'label': '60 months', 'value': 60},
                {'label': '84 months', 'value': 84},
                {'label': '120 months', 'value': 120},
                {'label': '180 months', 'value': 180},
                {'label': '240 months', 'value': 240},
                {'label': '300 months', 'value': 300},
                {'label': '360 months', 'value': 360},
                {'label': '480 months', 'value': 480}
            ],
            value=360
        ),
        html.Label("Credit History:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='credit_history',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1
        ),
        html.Label("Property Area:", style={'color': colors['text']}),
        dcc.Dropdown(
            id='property_area',
            options=[
                {'label': 'Rural', 'value': 'Rural'},
                {'label': 'Semiurban', 'value': 'Semiurban'},
                {'label': 'Urban', 'value': 'Urban'}
            ],
            value='Rural'
        ),
        html.Button('Check Eligibility', id='submit-val', n_clicks=0, style={'backgroundColor': colors['accent'], 'color': 'white'}),
        html.Div(id='output', style={'marginTop': '20px', 'fontWeight': 'bold'})
    ], style={'marginBottom': '20px', 'padding': '20px', 'borderRadius': '5px', 'backgroundColor': '#ffffff'})
])

# Define Callback Function for Predictions
@app.callback(
    Output('output', 'children'),
    Input('submit-val', 'n_clicks'),
    [Input('gender', 'value'),
        Input('married', 'value'),
        Input('dependents', 'value'),
        Input('education', 'value'),
        Input('self_employed', 'value'),
        Input('applicant_income', 'value'),
        Input('coapplicant_income', 'value'),
        Input('loan_amount', 'value'),
        Input('loan_amount_term', 'value'),
        Input('credit_history', 'value'),
        Input('property_area', 'value')]
)
def update_output(n_clicks, gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                    credit_history, property_area):
    if n_clicks > 0:
        # Calculate Total Income
        total_income = int(applicant_income) + int(coapplicant_income)

        # Calculate Log of Total Income
        total_income_log = np.log(total_income)

        # Calculate EMI (Equated Monthly Installment)
        emi = int(loan_amount) / int(loan_amount_term)

        # Calculate Income After EMI
        income_after_emi = total_income - (emi * 1000)  # Assuming EMI is in thousands

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'Total_Income': [total_income],
            'Total_Income_Log': [total_income_log],
            'EMI': [emi],
            'Income_After_EMI': [income_after_emi],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data)

        # Make predictions
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            return html.Div('Loan Approved', style={'color': 'green'})
        else:
            return html.Div('Loan Rejected', style={'color': 'red'})

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)