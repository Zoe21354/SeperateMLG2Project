import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import pickle

# Load the trained ML model
with open('Artifacts/Model_2.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Dash app
app = dash.Dash(__name__)

# Define CSS styles
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout the web page
app.layout = html.Div([
    html.H1("Loan Eligibility Prediction"),

    html.Label("Gender:"),
    dcc.Dropdown(
        id='gender-input',
        options=[
            {'label': 'Male', 'value': 1},
            {'label': 'Female', 'value': 0}
        ],
        value=1  # Default value
    ),

    html.Label("Married:"),
    dcc.Dropdown(
        id='married-input',
        options=[
            {'label': 'Yes', 'value': 1},
            {'label': 'No', 'value': 0}
        ],
        value=1  # Default value
    ),

    html.Label("Dependents:"),
    dcc.Input(id='dependents-input', type='number', value=0),

    html.Label("Education:"),
    dcc.Dropdown(
        id='education-input',
        options=[
            {'label': 'Graduate', 'value': 1},
            {'label': 'Non-Graduate', 'value': 0}
        ],
        value=1  # Default value
    ),

    html.Label("Self Employed:"),
    dcc.Dropdown(
        id='self-employed-input',
        options=[
            {'label': 'Yes', 'value': 1},
            {'label': 'No', 'value': 0}
        ],
        value=0  # Default value
    ),

    html.Label("Applicant's Income:"),
    dcc.Input(id='income-input', type='number', value=5000),

    html.Label("Coapplicant's Income:"),
    dcc.Input(id='co-income-input', type='number', value=0),

    html.Label("Loan Amount:"),
    dcc.Input(id='loan-amount-input', type='number', value=100000),

    html.Label("Loan Amount Term:"),
    dcc.Input(id='loan-term-input', type='number', value=360),

    html.Label("Credit History:"),
    dcc.Dropdown(
        id='credit-history-input',
        options=[
            {'label': 'Yes', 'value': 1},
            {'label': 'No', 'value': 0}
        ],
        value=1  # Default value
    ),

    html.Label("Property Area:"),
    dcc.Dropdown(
        id='property-area-input',
        options=[
            {'label': 'Rural', 'value': 'Rural'},
            {'label': 'Semiurban', 'value': 'Semiurban'},
            {'label': 'Urban', 'value': 'Urban'}
        ],
        value='Urban'  # Default value
    ),

    html.Button('Check Eligibility', id='submit-val', n_clicks=0),
    html.Div(id='output-state')
])

# Define callback function
@app.callback(
    Output('output-state', 'children'),
    [Input('submit-val', 'n_clicks')],
    [
        State('gender-input', 'value'),
        State('married-input', 'value'),
        State('dependents-input', 'value'),
        State('education-input', 'value'),
        State('self-employed-input', 'value'),
        State('income-input', 'value'),
        State('co-income-input', 'value'),
        State('loan-amount-input', 'value'),
        State('loan-term-input', 'value'),
        State('credit-history-input', 'value'),
        State('property-area-input', 'value')
    ]
)
def update_output(n_clicks, gender, married, dependents, education, self_employed,
                    income, co_income, loan_amount, loan_term, credit_history, property_area):
    if n_clicks > 0:
        # Prepare input data for prediction
        data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'Applicant_Income': [income],
            'Coapplicant_Income': [co_income],
            'Loan_Amount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area_Rural': [1 if property_area == 'Rural' else 0],
            'Property_Area_Semiurban': [1 if property_area == 'Semiurban' else 0],
            'Property_Area_Urban': [1 if property_area == 'Urban' else 0],
        }
        X_test = pd.DataFrame(data)
        # Perform prediction using the loaded model
        prediction = model.predict(X_test)
        if prediction[0] == 1:
            return html.Div('Congratulations! You are eligible for the loan.', style={'color': 'green'})
        else:
            return html.Div('Sorry, you are not eligible for the loan.', style={'color': 'red'})
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
