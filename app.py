from dash import dcc, html, Dash, Input, Output, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

data = pd.read_excel('all_adr_data.xlsx')
def get_info(ticker, data, disc, amt):
    ticker = ticker.upper()
    amt = amt/100

    ticker += ' EQUITY'
    data = data[data['adr'] == ticker][['date', 'premium', 'close_to_twap', 'absolute return']].dropna()
    if len(data) == 0:
        return data, ('Error: Invalid Ticker')
    if len(data) == 0:
        return data, ('Error: Invalid Premium/Discount')
    if disc:
        data = data[data['premium'] <= -amt]
        data.columns =  ['Date', 'Prem/Disc', 'Return', 'Absolute Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    else:
        data = data[data['premium'] >= amt]
        data.columns =  ['Date', 'Prem/Disc', 'Return', 'Absolute Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    int_list = np.where(np.sign(data['Return']) != np.sign(data['Prem/Disc']), 1, -1)

    data['Right Way Return'] = np.where(data['Absolute Return'] < 0, data['Absolute Return'], int_list * data['Absolute Return'])
    data = data.reset_index(drop = True)
    del data['Absolute Return']
    data['Return'] = data['Right Way Return']
    del data['Right Way Return']
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
    zdata = data.dropna().tail(10)
    amt = str(round(zdata['Return'].mean()*10000)) + "bps"
    acc = str(round(np.sign(zdata['Return']).replace(-1,0).mean()*100)) + "%"
    string = f"Last {len(zdata)} Right Way: {acc}, Last {len(zdata)} Avg Return: {amt}"
    data.columns = ['Date', 'Prem/Disc (%)', 'Return (%)']
    data['Prem/Disc (%)'] *=100
    data['Prem/Disc (%)'] = round(data['Prem/Disc (%)'],2)
    data['Return (%)']*=100
    data['Return (%)'] = round(data['Return (%)'],2)
    return data.sort_values(by = 'Date', ascending = False).dropna(), string

import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash import dash_table
from datetime import datetime

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
    dcc.Input(id='amt-input', placeholder='Enter % disc/prem (abs value)', type='number', style={'margin': '10px', 'width': '24%'}),
    html.Button('Enter', id='enter-button', n_clicks=0, style={'margin': '10px'}),
    html.Div([
        dcc.DatePickerSingle(
            id='start-date-picker',
            placeholder='Start Date',
            style={'margin': '10px', 'width': '100px'},  # Adjusted width
        ),
        dcc.DatePickerSingle(
            id='end-date-picker',
            placeholder='End Date',
            date=datetime.today().strftime('%Y-%m-%d'),  # Default to today's date
            style={'margin': '10px', 'width': '100px'},  # Adjusted width
        )
    ]),

    html.Div(id='result-table', style={'margin-top': '20px'})  # Placeholder for the styled DataFrame
], style={'padding': '30px'})

@app.callback(
    Output('result-table', 'children'),
    Input('enter-button', 'n_clicks'),
    [State('ticker-input', 'value'), State('disc-flag-input', 'value'), 
     State('amt-input', 'value'), State('start-date-picker', 'date'), 
     State('end-date-picker', 'date')]
)



def update_result_table(n_clicks, ticker, flags, amt_threshold, start_date, end_date):

    if n_clicks > 0:
        disc_flag = 'discount' in flags
        df, result_string = get_info(ticker, data, disc_flag, amt_threshold)
        if df.empty:
            return html.P('No data available for the given parameters.')

        # Find the maximum absolute value for the gradient scale.
        max_abs_value = df['Prem/Disc (%)'].abs().max()
        light_blue = [183, 226, 240]  # Darker than the previous light blue
        dark_blue = [65, 105, 225]    # Darker than the previous dark blue
        if start_date is not None:
            df = df[df['Date'] >= start_date]
        if end_date is not None:
            df = df[df['Date'] <= end_date]
        zdata = df.dropna().head(10)
        amt = str(round(zdata['Return (%)'].mean()*100)) + "bps"
        acc = str(round(np.sign(zdata['Return (%)']).replace(-1,0).mean()*100)) + "%"
        result_string = f"Last {len(zdata)} Right Way: {acc}, Last {len(zdata)} Avg Return: {amt}"
        data_filtered = data[data['adr'] == ticker.upper() + " EQUITY"].sort_values('date')
        last_date = data_filtered.iloc[0]['date']
        formatted_date = last_date.strftime('%m/%Y')
        result_string += f". Coverage for this ticker beginning {formatted_date}."
        # Find the maximum absolute value for the gradient scale.

        # Create a DataTable component to display the dataframe with conditional styling
        table = dash_table.DataTable(
            id='result-data',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records'),
            sort_action = "native",
            style_table={'height': '333px', 'overflowY': 'auto'},
            style_cell={'padding': '5px', 'fontSize': '14px'},
            page_size=120,  # Adjust based on preference
                    style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Return (%)} < 0',
                        'column_id': 'Return (%)'
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Return (%)} > 0',
                        'column_id': 'Return (%)'
                    },
                    'backgroundColor': '#2ECC40',
                    'color': 'white'
                }
            ] 
 + 
        [
            {
                'if': {
                    'filter_query': '{{Prem/Disc (%)}} = {}'.format(value),
                    'column_id': 'Prem/Disc (%)'
                },
                'backgroundColor': color_scale(abs(value), max_abs_value, 0, light_blue, dark_blue),
                'color': 'white' if abs(value) > max_abs_value * 0.5 else 'black'
            }
            for value in df['Prem/Disc (%)'].unique() if not pd.isnull(value)
        ]

        )
    
        return [html.P(result_string, style={'margin-bottom': '10px'}), table]
    else:
        return None

if __name__ == '__main__':
    app.run_server(debug=True)
