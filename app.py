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
        data.columns =  ['Date', 'Prem/Disc', 'Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    else:
        data = data[data['premium'] >= amt]
        data.columns =  ['Date', 'Prem/Disc', 'Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    if len(data) == 0:
        return data, ('Error: Invalid Premium/Discount')
    data['Return'] = round(data['Return'],3)
    data = data.reset_index(drop = True)
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
    if len(data) < 10:
        amt = str(round(data['Return'].mean()*10000)) + "bps"
        acc = str(np.sign(data['Return']).replace(-1,0).mean()*100) + "%"
    zdata = data.dropna().tail(10)
    amt = str(round(zdata['Return'].mean()*10000)) + "bps"
    acc = str(round(np.sign(zdata['Return']).replace(-1,0).mean()*100)) + "%"
    string = f"Last {len(zdata)} Right Way: {acc}, Last {len(zdata)} Avg Return: {amt}"
    if sortby == 'Disc/Prem':
        if disc == True:
            sortby = 'Prem/Disc'
        else:
            sortby = 'Prem/Disc'
    return data.sort_values(by = sortby, ascending = False).dropna(), string

import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash import dash_table

def color_scale(value, max_value, min_value, start_color, end_color):
    """
    Returns a color from a gradient scale based on the value's position between min_value and max_value.
    """
    if max_value == min_value:  # Avoid division by zero
        return f'rgb({start_color[0]}, {start_color[1]}, {start_color[2]})'
    normalized = (value - min_value) / (max_value - min_value)
    color = [int(start + (end - start) * normalized) for start, end in zip(start_color, end_color)]
    return f'rgb({color[0]}, {color[1]}, {color[2]})'

# Define your color range for the gradient here.


app = dash.Dash(__name__)
server = app.server

# Create input components
app.layout = html.Div([
    html.H1("ADR Info Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),
    dcc.Input(id='ticker-input', placeholder='Enter ADR symbol (i.e. BABA)', type='text', style={'margin': '10px', 'width': '20%'}),
    dcc.Dropdown(id='disc-flag-input', options=[
        {'label': 'Discount', 'value': 'discount'},
        {'label': 'Premium', 'value': 'premium'}
    ], placeholder='Select discount or premium', style={'margin': '10px', 'width': '50%'}),
    dcc.Input(id='amt-input', placeholder='Enter amount threshold (absolute value)', type='number', style={'margin': '10px', 'width': '24%'}),
    html.Button('Enter', id='enter-button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='result-table', style={'margin-top': '20px'})  # Placeholder for the styled DataFrame
], style={'padding': '30px'})

@app.callback(
    Output('result-table', 'children'),
    [Input('enter-button', 'n_clicks')],
    [State('ticker-input', 'value'), State('disc-flag-input', 'value'), State('amt-input', 'value')]
)


def update_result_table(n_clicks, ticker, flags, amt_threshold):

    if n_clicks > 0:
        disc_flag = 'discount' in flags
        df, result_string = get_info(ticker, data, disc_flag, amt_threshold)
        if df.empty:
            return html.P('No data available for the given parameters.')

        # Find the maximum absolute value for the gradient scale.
        max_abs_value = df['Prem/Disc'].abs().max()
        light_blue = [183, 226, 240]  # Darker than the previous light blue
        dark_blue = [65, 105, 225]    # Darker than the previous dark blue

        # Find the maximum absolute value for the gradient scale.

        # Create a DataTable component to display the dataframe with conditional styling
        table = dash_table.DataTable(
            id='result-data',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records'),
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'padding': '5px', 'fontSize': '14px'},
            page_size=100,  # Adjust based on preference
                    style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Return} < 0',
                        'column_id': 'Return'
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Return} > 0',
                        'column_id': 'Return'
                    },
                    'backgroundColor': '#2ECC40',
                    'color': 'white'
                }
            ] 
 + 
        [
                {
                    'if': {
                        'filter_query': f'{{{'Prem/Disc'}}} = {value}',
                        'column_id': 'Prem/Disc'
                    },
                    'backgroundColor': color_scale(abs(value), max_abs_value, 0, light_blue, dark_blue),
                    'color': 'white'
                }
                for value in df['Prem/Disc'].unique()
            ]
        )
    
        return [html.P(result_string, style={'margin-bottom': '10px'}), table]
    else:
        return None

if __name__ == '__main__':
    app.run_server(debug=True)
