import base64
from dash import dcc, html, Dash, Input, Output, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
api_key = 'PKAHFOAA86UB7M7PQB62'
api_secret = 'dEPAZPtNU3yBKiYXcihnhys2Tg6mIkwEm67tEgOV'
base_url = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
import tiingo
from tiingo import TiingoClient
config = {}
config['session'] = True
config['api_key'] = "7eef93d596bb5db06a125388ed2ae999a4332fd7"
client = TiingoClient(config)
import yfinance as yf
api_key2 = 'PKXY2KAIRXBONPE3U5HA'
api_secret2 = 'HdaXpW8p4FzyRZSGrhY99BOpgPcASdmGrXXNY0hR'
base_url = 'https://paper-api.alpaca.markets'
api2 = tradeapi.REST(api_key2, api_secret2, base_url, api_version='v2')

data = pd.read_excel('all_adr_data.xlsx')
def get_info(ticker, data, disc, amt):

    ticker = ticker.upper()
    amt = amt/100

    ticker += ' EQUITY'
    data = data[data['adr'] == ticker][['date', 'premium', 'close_to_twap', 'absolute return']].dropna()
    if len(data) == 0:
        return data, ('Error: Invalid Ticker')
    if disc:
        data = data[data['premium'] <= -amt]
        if len(data) == 0:
            return data, ('Error: Invalid Ticker')
        data.columns =  ['Date', 'Prem/Disc', 'Return', 'Absolute Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    else:
        data = data[data['premium'] >= amt]
        if len(data) == 0:
            return data, ('Error: Invalid Ticker')
        data.columns =  ['Date', 'Prem/Disc', 'Return', 'Absolute Return']
        data['Prem/Disc'] = round(data['Prem/Disc'], 3)
    int_list = np.where(np.sign(data['Return']) != np.sign(data['Prem/Disc']), 1, -1)
    data['Trade'] = np.where((data['Absolute Return'] > 0) & (data['Return'] > 0), 'Long',
                             np.where((data['Absolute Return'] > 0) & (data['Return'] < 0), 'Short', 'Zero'))
    data['Right Way Return'] = np.where(data['Absolute Return'] < 0, data['Absolute Return'], int_list * data['Absolute Return'])
    data = data.reset_index(drop = True)
    del data['Absolute Return']
    data['Return'] = data['Right Way Return']
    del data['Right Way Return']
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
    data.columns = ['Date', 'Prem/Disc (%)', 'Return (%)', 'Trade']

    data['Prem/Disc (%)'] *=100
    data['Prem/Disc (%)'] = round(data['Prem/Disc (%)'],2)
    data['Return (%)']*=100
    data['Return (%)'] = round(data['Return (%)'],2)
    return data.sort_values(by = 'Date', ascending = False).dropna()
def show_all(ticker, data, disc, amt):
    if amt != 0:
        return get_info(ticker, data, disc, amt)
    else:
        disc1 = get_info(ticker, data, True, 0)
        prem1 = get_info(ticker, data, False, 0)
        df = pd.concat([disc1, prem1]).reset_index(drop = True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by = 'Date', ascending = False).dropna()
        df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")
        return df 
def get_everything(ticker, amount, dailyhigh = 0):
    td = pd.to_datetime('today')
    start = td - timedelta(days = 365*8)
    try:
        df = client.get_dataframe(ticker, frequency='Daily', startDate= start, endDate= td - BDay(1))
    except:
        return pd.DataFrame()
    highs = []
    at = []
    lows = []
    blp = df.copy().reset_index()
    for i in range(0,len(blp)):
        dfi = blp.head(i)
        num = dfi[dfi['adjClose'] > blp['adjClose'][i]]
        numlow = dfi[dfi['adjClose'] < blp['adjClose'][i]]
        if len(num) == 0:
            highs.append(i - 1)
            at.append(True)       
        else:
            highs.append(i - num.index.tolist()[-1] - 1)
            at.append(False)
        if len(numlow) == 0:
            lows.append(i-1)
        else:
            lows.append(-1* (i - numlow.index.tolist()[-1] - 1))

    df['# Day High'] = highs
    df['l'] = lows
    df['# Day High'] = df['# Day High'].replace(0,np.nan)
    df['# Day High'] = df['# Day High'].fillna(df['l'])
    df['Close to Open'] = round(((df['adjOpen'].shift(-1) - df['adjClose']) / df['adjClose'] * 100).shift(1),3)
    df['Open to Close'] = round(((df['adjClose'] - df['adjOpen']) / df['adjOpen'] * 100).shift(0),3)
    if amount > 0:
        df = df[df['Close to Open'] >= amount]
    elif amount < 0:
        df = df[df['Close to Open'] <= amount]
    else:
        df = df.tail(30)
    if dailyhigh is None:
        dailyhigh = 0
    if dailyhigh > 0:
        df = df[df['# Day High'] >= dailyhigh]
    elif dailyhigh < 0:
        df = df[df['# Day High'] <= dailyhigh]


    def get_df(row):
        date_obj = Timestamp(row.name)
        eastern = timezone('US/Eastern')
        if date_obj.tzinfo is None:
            date_obj = eastern.localize(date_obj)
        else:
            date_obj = date_obj.tz_convert(eastern)
        dst_start = datetime(date_obj.year, 3, 8) + timedelta(days=6 - datetime(date_obj.year, 3, 8).weekday())
        dst_end = datetime(date_obj.year, 11, 1) + timedelta(days=6 - datetime(date_obj.year, 11, 1).weekday())
        dst_start = eastern.localize(dst_start)
        dst_end = eastern.localize(dst_end)
        if dst_start <= date_obj < dst_end:
            time_offset = "-04:00"
        else:
            time_offset = "-05:00"
        date_obj += BDay(1)
        start_time = date_obj.replace(hour=9, minute=30, second=0).isoformat()
        end_time = (date_obj + pd.offsets.BusinessDay(0)).replace(hour=11, minute=0, second=0).isoformat() 
        if np.random.choice([1,2]) ==1:
            zapi = api
        else:
            zapi = api2
        return zapi.get_bars(ticker, '1Min', start=start_time, end=end_time).df    

    def get_rets(nowdf, min):
        open = float(nowdf['open'][0])
        return round(100*(float(nowdf.head(min+1)['open'][-1]) - open)/open,3)

    all_dfs = [get_df(row) for index, row in df.iterrows()]

    # Calculate returns for each dataframe and store them in new columns in df
    df['1 Min Return'] = [get_rets(df, 1) for df in all_dfs]
    df['3 Min Return'] = [get_rets(df, 3) for df in all_dfs]
    df['5 Min Return'] = [get_rets(df, 5) for df in all_dfs]
    df['10 Min Return'] = [get_rets(df, 10) for df in all_dfs]
    df['15 Min Return'] = [get_rets(df, 15) for df in all_dfs]
    df = df[['Close to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return', 'Open to Close', '# Day High']]
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.iloc[::-1].reset_index()
    try:
        earnings_hist = pd.DataFrame(yf.Ticker(ticker).get_earnings_dates(limit = 40))
        earnings_hist['Earnings Date'] = earnings_hist.index
        earnings_hist_date = earnings_hist['Earnings Date'].dt.date
        df['Prev Day Earnings'] = df['date'].apply(
        lambda date: "Yes" if (pd.to_datetime(date) - BDay(1)).date() in earnings_hist_date.values else "No"
    )
    except:
        df['Prev Day Earnings'] = np.nan

    return df
def get_everything2(ticker, amount, weekday = "No Weekday Filter", dailyhigh = 0, offopen = "No Off Open Returns",  ath = "No All-Time High Filter"):
    td = pd.to_datetime('today')
    start = td - timedelta(days = 365*8)
    try:
        df = client.get_dataframe(ticker, frequency='Daily', startDate= start, endDate= td - BDay(1))
    except:
        return pd.DataFrame()
    highs = []
    at = []
    lows = []
    blp = df.copy().reset_index()
    for i in range(0,len(blp)):
        dfi = blp.head(i)
        num = dfi[dfi['adjClose'] > blp['adjClose'][i]]
        numlow = dfi[dfi['adjClose'] < blp['adjClose'][i]]
        if len(num) == 0:
            highs.append(i - 1)
            at.append(True)       
        else:
            highs.append(i - num.index.tolist()[-1] - 1)
            at.append(False)
        if len(numlow) == 0:
            lows.append(i-1)
        else:
            lows.append(-1* (i - numlow.index.tolist()[-1] - 1))

    df['# Day High'] = highs
    df['l'] = lows
    df['# Day High'] = df['# Day High'].replace(0,np.nan)
    df['# Day High'] = df['# Day High'].fillna(df['l'])
    del df['l']
    df['All Time High'] = at

    df['Prev Close to Close'] = round(100 * df['adjClose'].pct_change(),3)
    df['Close to Open'] = round(100*(df['adjOpen'].shift(-1) - df['adjClose'])/df['adjClose'],3)
    df['Open to Close'] = round(((df['adjClose'] - df['adjOpen']) / df['adjOpen'] * 100).shift(-1),3)
    if dailyhigh is None:
        dailyhigh = 0  
    if amount > 0:
        df = df[df['Prev Close to Close'] >= amount]
    elif amount < 0:
        df = df[df['Prev Close to Close'] <= amount]
    else:
        df = df.tail(30)
    if dailyhigh > 0:
        df = df[df['# Day High'] >= dailyhigh]
    elif dailyhigh < 0:
        df = df[df['# Day High'] <= dailyhigh]

    if ath != 'No All-Time High Filter':
            df = df[df['All Time High'] == True]

    def get_df(row):
        try:
            date_obj = Timestamp(row['date'])
            eastern = timezone('US/Eastern')
            if date_obj.tzinfo is None:
                date_obj = eastern.localize(date_obj)
            else:
                date_obj = date_obj.tz_convert(eastern)
            dst_start = datetime(date_obj.year, 3, 8) + timedelta(days=6 - datetime(date_obj.year, 3, 8).weekday())
            dst_end = datetime(date_obj.year, 11, 1) + timedelta(days=6 - datetime(date_obj.year, 11, 1).weekday())
            dst_start = eastern.localize(dst_start)
            dst_end = eastern.localize(dst_end)
            date_obj += BDay(1)
            start_time = date_obj.replace(hour=9, minute=30, second=0).isoformat()
            end_time = (date_obj + pd.offsets.BusinessDay(0)).replace(hour=11, minute=0, second=0).isoformat() 
            return api2.get_bars(ticker, '1Min', start=start_time, end=end_time).df     
        except:
            return pd.DataFrame()

    def get_rets(nowdf, min):
        if len(nowdf) == 0:
            return np.nan
        open = float(nowdf['open'][0])
        return round(100*(float(nowdf.head(min+1)['open'][-1]) - open)/open,3)

    df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index()
    df['Weekday'] = (pd.to_datetime(df['date'])+BDay(1)).dt.day_name()
    if weekday != 'No Weekday Filter':
        df = df[df['Weekday'] == weekday]
    try:
        earnings_hist = pd.DataFrame(yf.Ticker(ticker).get_earnings_dates(limit = 40))
        earnings_hist['Earnings Date'] = earnings_hist.index
        earnings_hist_date = earnings_hist['Earnings Date'].dt.date
        df['Prev Day Earnings'] = df['date'].apply(
        lambda date: "Yes" if (pd.to_datetime(date) - BDay(1)).date() in earnings_hist_date.values else "No"
    )
    except Exception as e:
        df['Prev Day Earnings'] = str(e)
    if offopen != 'No Off Open Returns':
        all_dfs = [get_df(row) for index, row in df.iterrows()]
        # Calculate returns for each dataframe and store them in new columns in df
        df['1 Min Return'] = [get_rets(df, 1) for df in all_dfs]
        df['3 Min Return'] = [get_rets(df, 3) for df in all_dfs]
        df['5 Min Return'] = [get_rets(df, 5) for df in all_dfs]
        df['10 Min Return'] = [get_rets(df, 10) for df in all_dfs]
        df['15 Min Return'] = [get_rets(df, 15) for df in all_dfs]
        df = df[['date', 'Prev Close to Close', 'Close to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return', 'Open to Close', '# Day High', 'All Time High', 'Prev Day Earnings', 'Weekday']]
        df = df.iloc[::-1]
        
        return df.dropna().reset_index(drop = True)

    else:
        df = df.iloc[::-1]
        return df[['date', 'Prev Close to Close', 'Close to Open', 'Open to Close', '# Day High', 'All Time High', 'Prev Day Earnings', 'Weekday']]

import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
from pandas import Timestamp
from pytz import timezone
import warnings
import io
warnings.filterwarnings('ignore')
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

import dash_auth
VALID_USERNAME_PASSWORD_PAIRS = {
    'merus' : '3ParkAvenue'
}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
server = app.server

# Create input components
app.layout = html.Div([
    html.H1("Arjun Dashboard", style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("ADR Lookback"),
    html.P("Visualize returns of ADR->ORD conversion trades for specific pairs and premium/discount levels"), 
    html.P("Prem/Disc is taken from snapshot around 3pm. Return is of 2hr twap in ORD Country out of right way position."),
    dcc.Input(id='ticker-input', placeholder='Enter ADR symbol (i.e. BABA)', type='text', style={'margin': '10px', 'width': '20%'}),
    dcc.Dropdown(id='disc-flag-input', options=[
        {'label': 'Discount', 'value': 'discount'},
        {'label': 'Premium', 'value': 'premium'}
    ], placeholder='Select discount or premium', style={'margin': '10px', 'width': '50%'}),
    dcc.Input(id='amt-input', placeholder='% Disc/Prem (abs value). (0 for no filter)', type='number', style={'margin': '10px', 'width': '27.6%'}),
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
    dbc.Button('Submit', id='enter-button', color = 'primary', n_clicks=0),
    html.Div(id='result-table', style={'margin-top': '20px'}),
    html.Hr(),
    html.H2("Off Open Return - Close to Open Gap Up"),
    html.P("Visualize returns off the open from when a stock gaps up or down (close->open return) a certain amount:"),
    dbc.Input(id='input-ticker', type='text', placeholder='Enter ticker, e.g., GME'),
    dbc.Input(id='input-amount', type='number', placeholder='Enter percent gap up, e.g., 50'),
    dcc.Input(id='input-high1', placeholder='Daily High Filter (0 For Default)', type='number', value = None, style={'margin': '10px', 'width': '29.6%'}),
    dbc.Button('Submit', id='submit-button', color='primary', n_clicks=0),
    dbc.Button("Download as Excel", id="download-button", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
    html.Div(id='output-table', style={'margin-top': '20px'}),
    dcc.Download(id="download-dataframe2-xlsx"),
    html.Hr(),
    html.H2("Overnight Return - Previous Close-Close Move"),
    html.P("Visualize returns off the open from when a stock has a significant close-close move the previous day:"),
    dbc.Input(id='input-tickers', type='text', placeholder='Enter ticker, e.g., TSLA'),
    dbc.Input(id='input-amounts', type='number', placeholder='Enter percent move observed yesterday, e.g., 6'),
    dcc.Dropdown(id='weekday-filter-dropdown', options=[
    {'label': 'No Weekday Filter', 'value': 'No Weekday Filter'},
    {'label': 'Monday', 'value': 'Monday'},
    {'label': 'Tuesday', 'value': 'Tuesday'},
    {'label': 'Wednesday', 'value': 'Wednesday'},
    {'label': 'Thursday', 'value': 'Thursday'},
    {'label': 'Friday', 'value': 'Friday'},
        ], value = 'No Weekday Filter', placeholder='Select Weekday Filter', style={'margin': '10px', 'width': '50%'}),
    dcc.Input(id='input-high', placeholder='Daily High Filter (0 For Default)', type='number', value = None, style={'margin': '10px', 'width': '29.6%'}),
    dcc.Dropdown(id='ath-dropdown', options=[
    {'label': 'No All-Time High Filter', 'value': 'No All-Time High Filter'},
    {'label': 'Yes All-Time High Filter', 'value': 'Yes All-Time High Filter'},
        ], value = 'No All-Time High Filter', placeholder='Select All-Time High Filter', style={'margin': '10px', 'width': '50%'}),
    dcc.Dropdown(id='offopen-dropdown', options=[
    {'label': 'No Off Open Returns', 'value': 'No Off Open Returns'},
    {'label': 'Yes Off Open Returns', 'value': 'Yes Off-Open Returns'},
        ], value = 'No Off Open Returns', placeholder='View Off-Open Returns', style={'margin': '10px', 'width': '50%'}),
    dbc.Button('Submit', id='submit-buttons', color='primary', n_clicks=0),
    dbc.Button("Download as Excel", id="download-button2", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
    html.Div(id='doutput-table', style={'margin-top': '20px'}),
    dcc.Download(id="download-dataframe3-xlsx"),

],)


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
        df = show_all(ticker, data, disc_flag, amt_threshold)
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
    
@app.callback(
Output('output-table', 'children'),
[Input('submit-button', 'n_clicks')],
[dash.dependencies.State('input-ticker', 'value'),
    dash.dependencies.State('input-amount', 'value'), dash.dependencies.State('input-high1', 'value')]
)
def update_output(n_clicks, ticker, amount, high1):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything(ticker, amount, high1)            
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column == 'date':
                    continue
                if column == 'Prev Day Earnings':
                    continue
                if column == '# Day High':
                    continue
                # Convert column values to numeric
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
                if column == '# Day High':
                    continue
                strin = "{" + column + "}"
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' > 0', 'column_id': column},
                    'backgroundColor': '#228C22 ', 'color':'white'
                })
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' < 0', 'column_id': column},
                    'backgroundColor': '#FF6666 ', 'color':'white'
                })

            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=8,
                style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditionals  # Apply color coding
            )
        except Exception as e:
            return html.Div(f"Error fetching data: {str(e)}")
    return html.Div("Submit a ticker and amount to see data.")
@app.callback(
    Output('download-dataframe2-xlsx', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('input-ticker', 'value'),
     State('input-amount', 'value')]
)
def generate_excel(n_clicks, ticker, amount):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything(ticker, amount)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.close()
            xlsx_data = output.getvalue()
            return dcc.send_bytes(xlsx_data, "output.xlsx")
        except Exception as e:
            return None
    return None
@app.callback(
    Output('doutput-table', 'children'),
    [Input('submit-buttons', 'n_clicks')],
    [State('input-tickers', 'value'),
     State('input-amounts', 'value'),
     State('weekday-filter-dropdown', 'value'),
     State('input-high', 'value'),
     State('offopen-dropdown', 'value'),
     State('ath-dropdown', 'value')]
)
def update_output(n_clicks, ticker, amount, weekday_filter, dailyhigh, offopen, ath):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything2(ticker, amount, weekday_filter, dailyhigh, offopen, ath)            
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column == 'date':
                    continue
                if column == 'Weekday':
                    continue 
                if column == '# Day High':
                    continue
                if column == 'All Time High':
                    continue
                if column == "Prev Day Earnings":
                    continue
                # Convert column values to numeric
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
                if column == 'All Time High':
                    continue
                if column == '# Day High':
                    continue
                if column == "Prev Day Earnings":
                    continue

                strin = "{" + column + "}"
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' > 0', 'column_id': column},
                    'backgroundColor': '#228C22 ', 'color':'white'
                })
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' < 0', 'column_id': column},
                    'backgroundColor': '#FF6666 ', 'color':'white'
                })
            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=8,
                style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditionals  # Apply color coding
            )
        except Exception as e:
            return html.Div(f"Error fetching data: {str(e)}")
    return html.Div("Submit a ticker and amount to see data.")
@app.callback(
    Output('download-dataframe3-xlsx', 'data'),
    [Input('download-button2', 'n_clicks')],
    [State('input-ticker', 'value'),
     State('input-amount', 'value'),
     State('weekday-filter-dropdown', 'value')]
)
def generate_excel(n_clicks, ticker, amount, weekday_filter):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything2(ticker, amount, weekday_filter)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.close()
            xlsx_data = output.getvalue()
            return dcc.send_bytes(xlsx_data, "output.xlsx")
        except Exception as e:
            return None
    return None
if __name__ == '__main__':
    app.run_server(debug=True, port = 8051)
