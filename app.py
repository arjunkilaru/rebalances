from dash import dcc, html, Dash, Input, Output, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

data = pd.read_excel('all_adr_data.xlsx')
def get_info(ticker, data, disc, amt, sortby = 'Date'):
    ticker += ' EQUITY'
    data = data[data['adr'] == ticker][['date', 'premium', 'close_to_twap']]
    if len(data) == 0:
        return data, ('Error: Invalid Ticker')
    if disc:
        data = data[data['premium'] <= -amt]
        data.columns =  ['Date', 'Discount', 'Return']
        data['Discount'] = round(data['Discount'], 3)
    else:
        data = data[data['premium'] >= amt]
        data.columns =  ['Date', 'Premium', 'Return']
        data['Premium'] = round(data['Premium'], 3)
    if len(data) == 0:
        return data, ('Error: Invalid Premium/Discount')
    data['Return'] = round(data['Return'],3)
    data = data.reset_index(drop = True)
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
    if len(data) < 10:
        amt = str(round(data['Return'].mean()*10000)) + "bps"
        acc = str(np.sign(data['Return']).replace(-1,0).mean()*100) + "%"
    zdata = data.tail(10)
    amt = str(round(zdata['Return'].mean()*10000)) + "bps"
    acc = str(round(np.sign(zdata['Return']).replace(-1,0).mean()*100)) + "%"
    string = f"Last {len(zdata)} Accuracy: {acc}, Last {len(zdata)} Return: {amt}"
    if sortby == 'Disc/Prem':
        if disc == True:
            sortby = 'Discount'
        else:
            sortby = 'Premium'
    return data.sort_values(by = sortby, ascending = True), string

get_info('RMD', data, True, 0.01)[0]

import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash import dash_table

app = dash.Dash(__name__)
server - app.server

# Create input components
app.layout = html.Div([
    html.H1("ADR Info Dashboard"),
    dcc.Input(id='ticker-input', placeholder='Enter ticker', type='text'),
    dcc.Checklist(id='disc-flag-input', options=[
        {'label': 'Discount', 'value': 'discount'},
        {'label': 'Premium', 'value': 'premium'}
    ]),
    dcc.Input(id='amt-input', placeholder='Enter amount threshold', type='number'),
    html.Button('Enter', id='enter-button', n_clicks=0),
    html.Div(id='result-table')  # Placeholder for displaying the dataframe
])

# Callback to update the result table
@app.callback(
    Output('result-table', 'children'),
    [Input('enter-button', 'n_clicks')],
    [State('ticker-input', 'value'), State('disc-flag-input', 'value'), State('amt-input', 'value')]
)
def update_result_table(n_clicks, ticker, flags, amt_threshold):
    if n_clicks > 0:
        disc_flag = 'discount' in flags
        df, result_string = get_info(ticker, data, disc_flag, amt_threshold)
        # Create a DataTable component to display the dataframe
        table = dash_table.DataTable(
            id='result-data',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records')
        )
        return [html.P(result_string), table]
    else:
        return None

if __name__ == '__main__':
    app.run_server(debug=True)

