# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

 # Load dataset
data = pd.read_csv('data/winequality-red.csv')
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first')
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
    # Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']


# Split the dat a into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create an instance of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Predict the labels of the test set
# y_pred = logreg_model.predict(X_test)


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div(
    className='container',
    children=[
        html.H1('CO544-2023 Lab 3: Wine Quality Prediction'),

        html.Div(
            className='row',
            children=[
                html.Div(
                    className='col-md-6',
                    children=[
                        html.H3('Exploratory Data Analysis'),
                        html.Div(
                            className='row',
                            children=[
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label('Feature 1 (X-axis)'),
                                        dcc.Dropdown(
                                            id='x_feature',
                                            options=[{'label': col, 'value': col} for col in data.columns],
                                            value=data.columns[0]
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label('Feature 2 (Y-axis)'),
                                        dcc.Dropdown(
                                            id='y_feature',
                                            options=[{'label': col, 'value': col} for col in data.columns],
                                            value=data.columns[1]
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id='correlation_plot'),
                    ],
                ),
                html.Div(
                    className='col-md-6',
                    children=[
                        html.H3("Wine Quality Prediction"),
                        html.Div(
                            className='row',
                            children=[
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label("Fixed Acidity"),
                                        dcc.Input(id='fixed_acidity', type='number', required=True),
                                    ],
                                ),
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label("Volatile Acidity"),
                                        dcc.Input(id='volatile_acidity', type='number', required=True),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className='row',
                            children=[
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label("Citric Acid"),
                                        dcc.Input(id='citric_acid', type='number', required=True),
                                    ],
                                ),
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Label("Residual Sugar"),
                                        dcc.Input(id='residual_sugar', type='number', required=True),
                                    ],
                                ),
                            ],
                        ),
                        # Add the remaining input fields here
                        html.Div(
                            className='row',
                            children=[
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.Button('Predict', id='predict-button', n_clicks=0),
                                    ],
                                ),
                                html.Div(
                                    className='col-md-6',
                                    children=[
                                        html.H4("Predicted Quality"),
                                        html.Div(id='prediction-output'),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# Define the callback to update the correlation plot
@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value'),
     dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature, color='quality')
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig


# Define the callback function to predict wine quality
@app.callback(
    dash.dependencies.Output(component_id='prediction-output', component_property='children'),
    [dash.dependencies.Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('fixed_acidity', 'value'),
     dash.dependencies.State('volatile_acidity', 'value'),
     dash.dependencies.State('citric_acid', 'value'),
     dash.dependencies.State('residual_sugar', 'value'),
     # Include the remaining input states here
     ]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar):
    # Create input features array for prediction
    input_features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar]

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = logreg_model.predict([input_features])[0]

    # Return the prediction
    if prediction == 1:
        return 'This wine is predicted to be good quality.'
    else:
        return 'This wine is predicted to be bad quality.'


if __name__ == '__main__':
    app.run_server(debug=False)
