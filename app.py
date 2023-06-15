from dash import dcc, html, Dash, Input, Output, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

def make_preds(country, currency, aex, asx, dax, fc50, ftse_epra, ftse_geis, ftse_uk, ndx, other, russel, sp, stoxx, tsm, vl, shares, days, bs, absv):
    start = pd.read_excel('RebalancesAll.xlsx')
    X1 = start.drop(['Unnamed: 0', 'Ticker', 'Date', '30Min Before Close', 'Close', 'Pct Change Into', 'Open Next Day', 'Pct Change Out of', 'Right Way In', 'Right Way Out'], axis=1).loc[:, (start != 0).any(axis=0)]
    X = pd.get_dummies(X1, columns = ['Country Name', 'Currency', 'Buy/Sell'])
    y_in = start['Right Way In']
    y_out = start['Right Way Out']

    best_params = {'n_estimators': 101,
 'learning_rate': 0.011631605344793232,
 'max_depth': 3,
 'min_samples_split': 8,
 'min_samples_leaf': 4}
    
    best_params2 = {'n_estimators': 167,
 'learning_rate': 0.0028840860187493446,
 'max_depth': 4,
 'min_samples_split': 9,
 'min_samples_leaf': 7}
    
    
    clf1 = GradientBoostingClassifier(**best_params)
    clf2 = GradientBoostingClassifier(**best_params2)


    clf1.fit(X, y_in)
    clf2.fit(X, y_out)

    new_row = pd.Series({'Country Name': country, 'Currency': currency, 'AEX' : aex, 'ASX': asx, 'DAX' : dax, 'FC50': fc50, 'FTSE_EPRA' : ftse_epra, 
                         'FTSE_GEIS' : ftse_geis, 'FTSE_UK' :ftse_uk, 'NDX' : ndx, 'Other Index Events' : other, 'Russell': russel, 'S&P' :sp, 'STOXX': stoxx,
                           'TSM' :tsm, 'Value To Trade (m USD)' : vl, 'Shares To Trade (m)' : shares, 'Days To Trade' : days, 'Buy/Sell' : bs, 'Abs Value To Trade (m USD)' : absv})
    X1 = pd.concat([X1, pd.DataFrame([new_row])])

    now = pd.get_dummies(X1, columns = ['Country Name', 'Currency', 'Buy/Sell'])

    finalpred = now.iloc[-1].dropna()
    finalpred = pd.DataFrame(finalpred).T
    probs = pd.DataFrame()
    ps = [str(clf1.predict(finalpred)) + " Into Close", str(clf2.predict(finalpred)) + " Out of Close]
    ls = [np.max(clf1.predict_proba(finalpred)), np.max(clf2.predict_proba(finalpred))]
    probs.index = ['Into Print', 'Out of Print']
    probs['Right Way'] = ps
    probs['Confidence Level'] = ls
    return probs

app = Dash(__name__)
server = app.server


app.layout = html.Div(children=[
    html.H1(children='Rebalance Predictor (it will take ~30secs to make prediction)'),
    html.H2("This program will predict Right/Way wrong way moves for index events, using the last 2 rebalances as reference. Just enter in the details of the rebalance (ticker does not matter), including the number of shares bought or sold (input as negative) on each of the indexes. Put 0 if no trades are occuring on that index. Days to trade means the number of average day volumes being traded.", style={'font-size': '19px', 'font-weight': 'normal'}),

    html.Div(children=[
        html.Div(children=[
            html.Label('Country:'),
            dcc.Dropdown(
        id='country',
        options=[
            {'label': 'United States', 'value': 'United States'},
            {'label': 'United Kingdom', 'value': 'United Kingdom'},
            {'label': 'Germany', 'value': 'Germany'},
            {'label': 'France', 'value': 'France'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Netherlands', 'value': 'Netherlands'},
            {'label': 'Sweden', 'value': 'Sweden'},
            {'label': 'Australia', 'value': 'Australia'},
            {'label': 'Spain', 'value': 'Spain'},
            {'label': 'India', 'value': 'India'},
            {'label': 'Korea', 'value': 'Korea'},
            {'label': 'Israel', 'value': 'Israel'},
            {'label': 'Hong Kong', 'value': 'Hong Kong'},
            {'label': 'Ireland', 'value': 'Ireland'},
        ],
        value='United States'  # Default value
        ),      
        ]),


        html.Div(children=[
            html.Label('Currency:'),
        dcc.Dropdown(
            id='currency',
            options=[
                {'label': 'USD', 'value': 'USD'},
                {'label': 'GBP', 'value': 'GBP'},
                {'label': 'EUR', 'value': 'EUR'},
                {'label': 'HKD', 'value': 'HKD'},
                {'label': 'SEK', 'value': 'SEK'},
                {'label': 'AUD', 'value': 'AUD'},
                {'label': 'INR', 'value': 'INR'},
                {'label': 'KRW', 'value': 'KRW'},
            ],
            value='USD'  # Default value
        ),
        ]),

        html.Div(children=[
            html.Label('AEX:'),
            dcc.Input(id='aex', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('ASX:'),
            dcc.Input(id='asx', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('DAX:'),
            dcc.Input(id='dax', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('FC50:'),
            dcc.Input(id='fc50', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('FTSE_EPRA:'),
            dcc.Input(id='ftse_epra', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('FTSE_GEIS:'),
            dcc.Input(id='ftse_geis', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('FTSE_UK:'),
            dcc.Input(id='ftse_uk', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('NDX:'),
            dcc.Input(id='ndx', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Other Index Events:'),
            dcc.Input(id='other', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Russell:'),
            dcc.Input(id='russel', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('S&P:'),
            dcc.Input(id='sp', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('STOXX:'),
            dcc.Input(id='stoxx', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('TSM:'),
            dcc.Input(id='tsm', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Value To Trade (m USD):'),
            dcc.Input(id='vl', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Shares To Trade (m):'),
            dcc.Input(id='shares', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Days To Trade:'),
            dcc.Input(id='days', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Label('Buy/Sell:'),
            dcc.Dropdown(
                id='bs',
                options=[
                    {'label': 'Buy', 'value': 'Buy'},
                    {'label': 'Sell', 'value': 'Sell'},
                ],
                value='Buy'  # Default value
            ),
        ]),

        html.Div(children=[
            html.Label('Abs Value To Trade (m USD):'),
            dcc.Input(id='absv', type='number', value = 0),
        ]),

        html.Div(children=[
            html.Button('Submit', id='submit-button', n_clicks=0),
        ]),
    ]),

    html.Div(id='prediction-output'),
])


@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('country', 'value'),
     State('currency', 'value'),
     State('aex', 'value'),
     State('asx', 'value'),
     State('dax', 'value'),
     State('fc50', 'value'),
     State('ftse_epra', 'value'),
     State('ftse_geis', 'value'),
     State('ftse_uk', 'value'),
     State('ndx', 'value'),
     State('other', 'value'),
     State('russel', 'value'),
     State('sp', 'value'),
     State('stoxx', 'value'),
     State('tsm', 'value'),
     State('vl', 'value'),
     State('shares', 'value'),
     State('days', 'value'),
     State('bs', 'value'),
     State('absv', 'value')]
)
def update_output(n_clicks, country, currency, aex, asx, dax, fc50, ftse_epra, ftse_geis, ftse_uk, ndx, other, russel, sp, stoxx, tsm, vl, shares, days, bs, absv):
    if n_clicks > 0:
        try:
            df = make_preds(country, currency, aex, asx, dax, fc50, ftse_epra, ftse_geis, ftse_uk, ndx, other, russel, sp, stoxx, tsm, vl, shares, days, bs, absv)
            return html.Table(
                # Header
                [html.Tr([html.Th(col) for col in df.columns])] +
                # Body
                [html.Tr([
                    html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(len(df))]
            )
        except Exception as e:
            return html.Div(f'An error occurred: {str(e)}')
    return None
if __name__ == '__main__':
    app.run_server(debug=True)
