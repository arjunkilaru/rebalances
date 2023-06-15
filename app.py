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
    y_in = start[['Right Way In']]
    y_out = start[['Right Way Out']]

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
    ps = [clf1.predict(finalpred), clf2.predict(finalpred)]
    ls = [np.max(clf1.predict_proba(finalpred)), np.max(clf2.predict_proba(finalpred))]
    probs.index = ['Into Print', 'Out of Print']
    probs['Right Way'] = ps
    probs['Confidence Level'] = ls
    return probs

app = Dash(__name__)
server = app.server


app.layout = html.Div(children=[
    html.H1(children='Predictor'),

    html.Div(children=[
        html.Div(children=[
            html.Label('Country:'),
            dcc.Input(id='country', type='text'),
        ]),

        html.Div(children=[
            html.Label('Currency:'),
            dcc.Input(id='currency', type='text'),
        ]),

        html.Div(children=[
            html.Label('AEX:'),
            dcc.Input(id='aex', type='number'),
        ]),

        html.Div(children=[
            html.Label('ASX:'),
            dcc.Input(id='asx', type='number'),
        ]),

        html.Div(children=[
            html.Label('DAX:'),
            dcc.Input(id='dax', type='number'),
        ]),

        html.Div(children=[
            html.Label('FC50:'),
            dcc.Input(id='fc50', type='number'),
        ]),

        html.Div(children=[
            html.Label('FTSE_EPRA:'),
            dcc.Input(id='ftse_epra', type='number'),
        ]),

        html.Div(children=[
            html.Label('FTSE_GEIS:'),
            dcc.Input(id='ftse_geis', type='number'),
        ]),

        html.Div(children=[
            html.Label('FTSE_UK:'),
            dcc.Input(id='ftse_uk', type='number'),
        ]),

        html.Div(children=[
            html.Label('NDX:'),
            dcc.Input(id='ndx', type='number'),
        ]),

        html.Div(children=[
            html.Label('Other Index Events:'),
            dcc.Input(id='other', type='number'),
        ]),

        html.Div(children=[
            html.Label('Russell:'),
            dcc.Input(id='russel', type='number'),
        ]),

        html.Div(children=[
            html.Label('S&P:'),
            dcc.Input(id='sp', type='number'),
        ]),

        html.Div(children=[
            html.Label('STOXX:'),
            dcc.Input(id='stoxx', type='number'),
        ]),

        html.Div(children=[
            html.Label('TSM:'),
            dcc.Input(id='tsm', type='number'),
        ]),

        html.Div(children=[
            html.Label('Value To Trade (m USD):'),
            dcc.Input(id='vl', type='number'),
        ]),

        html.Div(children=[
            html.Label('Shares To Trade (m):'),
            dcc.Input(id='shares', type='number'),
        ]),

        html.Div(children=[
            html.Label('Days To Trade:'),
            dcc.Input(id='days', type='number'),
        ]),

        html.Div(children=[
            html.Label('Buy/Sell:'),
            dcc.Input(id='bs', type='text'),
        ]),

        html.Div(children=[
            html.Label('Abs Value To Trade (m USD):'),
            dcc.Input(id='absv', type='number'),
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
