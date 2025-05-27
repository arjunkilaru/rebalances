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
from pytz import timezone
import numba
import yfinance as yf
api_key2 = 'PKXY2KAIRXBONPE3U5HA'
api_secret2 = 'HdaXpW8p4FzyRZSGrhY99BOpgPcASdmGrXXNY0hR'
base_url = 'https://paper-api.alpaca.markets'
api2 = tradeapi.REST(api_key2, api_secret2, base_url, api_version='v2')

api_key3 = 'PKVTHPR9MG1N0OU7NADW'
api_secret3 = 'xR6Wsy8ZBEOMEq3Fb7DEwTVHva4pFsXzwMGzd3Du'
base_url = 'https://paper-api.alpaca.markets'
api3 = tradeapi.REST(api_key3, api_secret3, base_url, api_version='v2')
fy = pd.read_excel("JPM_DAILY_EOD_REBAL_COMPLETE.xlsx")
prices = pd.read_excel('full_jpm_db_with_price_action.xlsx')
fy = pd.merge(fy, prices, on = fy.columns.tolist(), how = 'outer')
fyu = pd.read_excel("JPM_UPCOMING_INDEX_EVENTS.xlsx")
edf = pd.read_excel('All_earnings_31725.xlsx')

# Define a function to assign values to the 'change' column

# Define a function to assign values to the 'change' column
def determine_change(row):
    if row['Share_Inc'] == True and row['Float_Inc'] == True:
        return 'S Inc & F Inc'
    elif row['Spin_Off'] == True and row['Add'] == True:
        return 'Spin Off Add'
    elif row['Spin_Off'] == True and row['Delete'] == True:
        return 'Spin Off Delete'
    elif row['Spin_Off'] == True:
        return 'Spin Off'
    elif row['Add'] == True:
        return 'Add'
    elif row['Share_Dec'] == True and row['Float_Dec'] == True:
        return 'S Dec & F Dec'
    elif row['Delete'] == True:
        return 'Delete'
    elif row['Country_change'] == True:
        return 'Country Change'
    elif row['Share_Inc'] == True:
        return 'S Inc'
    elif row['Float_Inc'] == True:
        return 'F Inc'
    elif row['Share_Dec'] == True:
        return 'S Dec'
    elif row['Float_Dec'] == True:
        return 'F Dec'
    else:
        return row['Change']

# Apply the function to each row to update the 'change' column
fy['Change'] = fy.apply(determine_change, axis=1)
fyu['Change'] = fyu.apply(determine_change, axis=1)

columns_to_drop = ['Share_Inc', 'Share_Dec', 'Float_Inc', 'Float_Dec', 'Add', 'Delete', 'Spin_Off', 'Country_change', 'Other']

# Drop the columns
fy.drop(columns=columns_to_drop, inplace=True)
fyu.drop(columns=columns_to_drop, inplace=True)

def stdevs(ticker, data, disc, amt):
    if disc:
        amt = amt*-1
    data = data[data['adr'] == ticker][['date', 'premium']].dropna().drop_duplicates().drop_duplicates(subset = 'date')
    mn = data['premium'].mean()
    std = data['premium'].std()
    if amt > mn:
        return str(round((amt - mn) / std, 3))
    else:
        return str(round(-1 * (amt - mn) / std, 3))


del fy['As_of_date']
del fyu['As_of_date']

for col in ["prev_close_to_close", "open_to_close", "close_to_next_open", "next_open_to_next_close", "annc_close_to_eff_close", "annc_close_to_next_open", "annc_next_open_to_next_close"]:
    fy[col] = round(100*fy[col],3)
    fyu[col]= np.nan

fy.columns = ['Region', 'Ticker', 'Company', 'Effective', 'Status', 'Index Name', 'Index Change', 'Weight Change', 'Value (mm)', 'Shares (mm)', 'ADV', 'Net Value (mm)', 'Net Shares (mm)', 'Net ADV', 'Announcement Date', 'Details', 'Prev Close to Effective Close', 'Effective Open to Close', 'Effective Close to Next Open', 'Effective T+1 Open to Close', 'Annc Close to Eff. Close', 'Annc Close to Next Open', 'Annc T+1 Open to Close']
fyu.columns = ['Region', 'Ticker', 'Company', 'Effective', 'Status', 'Index Name', 'Index Change', 'Weight Change', 'Value (mm)', 'Shares (mm)', 'ADV', 'Net Value (mm)', 'Net Shares (mm)', 'Net ADV', 'Announcement Date', 'Details', 'Prev Close to Effective Close', 'Effective Open to Close', 'Effective Close to Next Open', 'Effective T+1 Open to Close', 'Annc Close to Eff. Close', 'Annc Close to Next Open', 'Annc T+1 Open to Close']
data = pd.read_excel('all_adr_data.xlsx')

def rsi(ticker, days, rs):
    today = datetime.today()
    today = today.replace(hour=16, minute=0, second=0, microsecond=0)

    # Calculate the date three years before today
    three_years_ago = today - timedelta(days=365*3)

    # Convert to ISO format for API call (without fractional seconds)
    start_time = three_years_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = today.strftime('%Y-%m-%dT%H:%M:%SZ')
    blip = np.random.choice([1,2,3])
    if blip == 1:
        zapi = api
    elif blip == 2:
        zapi = api2
    else:
        zapi = api3
    # Fetch the bar data with a single API call
    #bars = zapi.get_bars(ticker, '15Min', start=start_time, end=end_time).df
    bars2 = client.get_dataframe(ticker, frequency='Daily', startDate= start_time, endDate= end_time)
    bars2['Close to Open'] = bars2['adjOpen'].shift(-1)
    bars2['Close to Open'] = round(100*(bars2['Close to Open'] - bars2['adjClose'])/bars2['adjClose'],3)
    bars2['1dayret'] = bars2['adjClose'].shift(-1)
    bars2['1dayret'] = round(100*(bars2['1dayret'] - bars2['adjClose'])/bars2['adjClose'],3)
    bars2['2dayret'] = bars2['adjClose'].shift(-2)
    bars2['2dayret'] = round(100*(bars2['2dayret'] - bars2['adjClose'])/bars2['adjClose'],3)
    bars2['1weekret'] = bars2['adjClose'].shift(-5)
    bars2['1weekret'] = round(100*(bars2['1weekret'] - bars2['adjClose'])/bars2['adjClose'],3)
    bars2['2weekret'] = bars2['adjClose'].shift(-10)
    bars2['2weekret'] = round(100*(bars2['2weekret'] - bars2['adjClose'])/bars2['adjClose'],3)
    bars2['diff'] = bars2['adjClose'].diff(1)
    bars2['gain'] = bars2['diff'].clip(lower=0).round(2)
    bars2['loss'] = bars2['diff'].clip(upper=0).abs().round(2)
    bars2['avg_gain'] = bars2['gain'].rolling(window=days, min_periods=days).mean()[:days+1]
    bars2['avg_loss'] = bars2['loss'].rolling(window=days, min_periods=days).mean()[:days+1]
    for i, row in enumerate(bars2['avg_gain'].iloc[days+1:]):
        bars2['avg_gain'].iloc[i + days + 1] =\
            (bars2['avg_gain'].iloc[i + days] *
            (days - 1) +
            bars2['gain'].iloc[i + days + 1])\
            / days
    # Average Losses
    for i, row in enumerate(bars2['avg_loss'].iloc[days+1:]):
        bars2['avg_loss'].iloc[i + days + 1] =\
            (bars2['avg_loss'].iloc[i + days] *
            (days - 1) +
            bars2['loss'].iloc[i + days + 1])\
            / days
    bars2['rs'] = bars2['avg_gain'] / bars2['avg_loss']
    bars2['rsi'] = round(100 - (100 / (1.0 + bars2['rs'])),3)
    bars2 = bars2[['adjClose', 'rsi', 'Close to Open', '1dayret', '2dayret', '1weekret', '2weekret']]
    if rs > 0:
        bars2 = bars2[bars2['rsi'] >= rs]
    elif rs < 0:
        bars2 = bars2[bars2['rsi'] <= -1*rs]
    else:
        bars2 = bars2.head(50)
    bars2.columns = ['Close Price', 'RSI', 'Close to Next Open', 'Close to Next Close', 'T+2', 'T + 1 Week', 'T + 2 Week']
    a =  bars2.iloc[::-1].reset_index()
    a['date'] = a['date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    return a

def returns(ticker, amount, time_str, open = 'open'):
    today = datetime.today() -BDay(1)
    today = today.replace(hour=16, minute=0, second=0, microsecond=0)

    # Calculate the date three years before today
    three_years_ago = today - timedelta(days=365*3)

    # Convert to ISO format for API call (without fractional seconds)
    start_time = three_years_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = today.strftime('%Y-%m-%dT%H:%M:%SZ')
    blip = np.random.choice([1,2,3])
    if blip == 1:
        zapi = api
    elif blip == 2:
        zapi = api2
    else:
        zapi = api3
    # Fetch the bar data with a single API call
    bars = zapi.get_bars(ticker, '15Min', start=start_time, end=end_time).df
    bars2 = client.get_dataframe(ticker, frequency='Daily', startDate= start_time, endDate= end_time)
    # Set timezone to Eastern Time
    eastern = timezone('US/Eastern')

    # Convert the index to the Eastern timezone
    bars.index = bars.index.tz_convert(eastern)

    # Filter the data to include only normal market hours (9:30 AM to 4:00 PM ET)
    market_open = datetime.strptime('09:30', '%H:%M').time()
    market_close = datetime.strptime('16:00', '%H:%M').time()
    bars = bars.between_time(market_open, market_close)
    specified_time = datetime.strptime(time_str, '%H:%M').time()

    # Add 15 minutes to the specified time
    bars['Return to 15 Min'] = round(100*bars['open'].pct_change().shift(-1),3)

    # Filter the DataFrame based on the updated time
    bars = bars[bars.index.time <= specified_time]
    zs, zx = bars, bars2
    b=zs
    b.index = pd.to_datetime(b.index)

    zs['day'] = zs.index.date
    dates = pd.concat([zx[zx['divCash'] != 0], zx[zx['splitFactor'] != 1]])
    zx['day'] = zx.index.date
    zs = zs[~zs['day'].isin(dates.index.date)]
    zx['yesterday close'] = zx['close'].shift(1)
    zx['next open'] = zx['open'].shift(-1)
    zx = zx[['close', 'open', 'yesterday close', 'next open', 'day']]
    zx.columns = ['today close', 'today open', 'yesterday close', 'next open', 'day']
    a = pd.merge(zs, zx, on='day', how='left')
    a.index = zs.index
    a = a.dropna()
    a['intraday return fc'] = round(100*(a['open'] - a['yesterday close'] ) /a['yesterday close'],3)
    a['return to close'] = round(100* (a['today close'] - a['open']) / a['open'],3)
    a['intraday return fo'] = round(100*(a['open'] - a['today open'] ) /a['today open'],3)
    a['return to next open'] = round(100*(a['next open'] - a['today close'])/a['today close'], 3)
    a = a.groupby('day').tail(1)
    if open.lower() == 'open':
        a = a[['intraday return fo', 'Return to 15 Min', 'return to close', 'return to next open']]
        a.columns = ['Return from Open', 'Return to 15 Min', 'Return to Close', 'Return to Next Open']
        if amount > 0:
            a = a[a['Return from Open'] >= amount]
        elif amount < 0:
            a = a[a['Return from Open'] <= amount]
        else:
            a = a.tail(20)
    else:
        a = a[['intraday return fc', 'Return to 15 Min', 'return to close', 'return to next open']]
        a.columns = ['Return from Prev Close', 'Return to 15 Min', 'Return to Close', 'Return to Next Open']
        if amount > 0:
            a = a[a['Return from Prev Close'] >= amount]
        elif amount <0:
            a = a[a['Return from Prev Close'] <= amount]
        else:
            a = a.tail(20)
    a = a.reset_index()
    return a.iloc[::-1]

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
    
import yfinance as yf

def get_earnings(ticker, amount):
    td = pd.to_datetime('today')
    start = td - timedelta(days = 365*8)
    try:
        df = client.get_dataframe(ticker, frequency='Daily', startDate= start, endDate= td - BDay(1))
        a = df
    except:
        return pd.DataFrame()
    earnings_hist = edf[edf['Unnamed: 0'] == ticker + ' US EQUITY']
    earnings_hist['Earnings Date'] = pd.to_datetime(edf['announcement_date'])	
    earnings_hist_date = earnings_hist['Earnings Date'].dt.date
    df = df.reset_index()
    df['Prev Day Earnings'] = df['date'].apply(
    lambda date: "Yes" if (pd.to_datetime(date) - BDay(1)).date() in earnings_hist_date.values else "No"
)
    df['Close to Open'] = round(((df['adjOpen'].shift(-1) - df['adjClose']) / df['adjClose'] * 100).shift(1),3)
    df['Open to Close'] = round(((df['adjClose'] - df['adjOpen']) / df['adjOpen'] * 100).shift(0),3)
    df = df[df['Prev Day Earnings'] == 'Yes']
    if amount > 0:
        df = df[df['Close to Open'] >= amount]
    elif amount < 0:
        df = df[df['Close to Open'] <= amount]
    else:
        df = df.tail(30)
    df = df[df['Prev Day Earnings'] == 'Yes']
    df = df.set_index('date')
    df['Earnings Day Return'] = df['Open to Close']
    df = df[['adjClose', 'Close to Open', 'Earnings Day Return']]
    df.columns = ['adjClose', 'Earnings Day Gap Up', 'Earnings Day Open to Close']
    def get_dfs(row, time):
        x = (row.name + BDay(time))
        a1 = (a[a.index == x])
        if len(a1) == 0:
            a1 = (a[a.index == x+BDay(1)])
        if len(a1) >0:
            return a1['adjClose'].iloc[0]
        else:
            return np.nan
    df['Earnings Day Gap Up'] = round(df['Earnings Day Gap Up'], 3)
    df['Earnings Day Open to Close'] = round(df['Earnings Day Open to Close'], 3)
    df['1 Week Return'] = round(100*((df.apply(lambda x: get_dfs(x, 5), axis = 1)) - df['adjClose']) / df['adjClose'],3)
    df['2 Week Return'] = round(100*((df.apply(lambda x: get_dfs(x, 10), axis = 1)) - df['adjClose']) / df['adjClose'],3)
    df['1 Month Return'] = round(100*((df.apply(lambda x: get_dfs(x, 20), axis = 1)) - df['adjClose']) / df['adjClose'],3)
    df['2 Month Return'] = round(100*((df.apply(lambda x: get_dfs(x, 40), axis = 1)) - df['adjClose']) / df['adjClose'],3)
    df['3 Month Return'] = round(100*((df.apply(lambda x: get_dfs(x, 60), axis = 1)) - df['adjClose']) / df['adjClose'],3)
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index().iloc[::-1]
    del df['adjClose']
    return df

def get_everything(ticker, amount, dailyhigh = 0, consq = 0, weekday = "No Weekday Filter", prevday = 0, rsi = 0):
    if rsi is None:
        rsi = 0
    def rsi2(ticker, days):
        today = datetime.today()
        today = today.replace(hour=16, minute=0, second=0, microsecond=0)

        # Calculate the date three years before today
        three_years_ago = today - timedelta(days=365*8)

        # Convert to ISO format for API call (without fractional seconds)
        start_time = three_years_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = today.strftime('%Y-%m-%dT%H:%M:%SZ')
        blip = np.random.choice([1,2,3])
        if blip == 1:
            zapi = api
        elif blip == 2:
            zapi = api2
        else:
            zapi = api3
        # Fetch the bar data with a single API call
        #bars = zapi.get_bars(ticker, '15Min', start=start_time, end=end_time).df
        bars2 = client.get_dataframe(ticker, frequency='Daily', startDate= start_time, endDate= end_time)
        bars2['Close to Open'] = bars2['adjOpen'].shift(-1)
        bars2['Close to Open'] = round(100*(bars2['Close to Open'] - bars2['adjClose'])/bars2['adjClose'],3)
        bars2['diff'] = bars2['adjClose'].diff(1)
        bars2['gain'] = bars2['diff'].clip(lower=0).round(2)
        bars2['loss'] = bars2['diff'].clip(upper=0).abs().round(2)
        bars2['avg_gain'] = bars2['gain'].rolling(window=days, min_periods=days).mean()[:days+1]
        bars2['avg_loss'] = bars2['loss'].rolling(window=days, min_periods=days).mean()[:days+1]
        delta = bars2['adjClose'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/days, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/days, adjust=False).mean()
        bars2['rsi'] = 100 - 100/(1 + avg_gain/avg_loss)
        bars2 = bars2[['adjClose', 'rsi']].dropna()
        a =  bars2.reset_index()
        a['date'] = a['date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
        return a
    td = pd.to_datetime('today')
    start = td - timedelta(days = 365*8)
    try:
        df = client.get_dataframe(ticker, frequency='Daily', startDate= start, endDate= td - BDay(1))
    except:
        return pd.DataFrame()
    def compute_days(arr):
        n = arr.shape[0]
        highs = np.empty(n, np.int64)
        lows = np.empty(n, np.int64)
        for i in range(n):
            last_hi = -1
            last_lo = -1
            for j in range(i):
                if arr[j] > arr[i]:
                    last_hi = j
                if arr[j] < arr[i]:
                    last_lo = j
            # days since last higher
            if last_hi == -1:
                highs[i] = i - 1
            else:
                highs[i] = i - last_hi - 1
            # days since last lower (negative)
            if last_lo == -1:
                lows[i] = -(i - 1)
            else:
                lows[i] = - (i - last_lo - 1)
        return highs, lows

    # usage inside get_everything, replacing the Python loop:
    arr = df['adjClose'].values
    highs, lows = compute_days(arr)
    df['# Day High'] = highs
    df['l'] = lows
    # then your existing replace/fill logic
    df['# Day High'].replace(0, np.nan, inplace=True)
    df['# Day High'].fillna(df['l'], inplace=True)
    df['Prev Day Return'] = round(100*df['adjClose'].pct_change(),3).shift(1)
    df['Close to Open'] = round(((df['adjOpen'].shift(-1) - df['adjClose']) / df['adjClose'] * 100).shift(1),3)
    df['Open to Close'] = round(((df['adjClose'] - df['adjOpen']) / df['adjOpen'] * 100).shift(0),3)
    df['Price Change'] = df['adjClose'].diff()
    df['Price Change'] = df['Price Change'].apply(lambda x: 1 if x>0 else -1)
    df['Consecutive Up/Down Days'] = df['Price Change'].groupby((df['Price Change'] != df['Price Change'].shift()).cumsum()).cumsum()
    df['Consecutive Up/Down Days'] = df['Consecutive Up/Down Days'].shift(1)
    df['date'] = df.index
    df['Weekday'] = (pd.to_datetime(df['date'])).dt.day_name()
    rsdf = rsi2(ticker, 3)
    df = df.iloc[1:]
    df = df.tail(len(rsdf))
    df['3D RSI'] = rsdf['rsi'].tolist()
    df['3D RSI'] = round(df['3D RSI'].shift(1),2)
    if prevday is None:
        prevday = 0
    if prevday > 0:
        df = df[df['Prev Day Return'] >= prevday]
    if prevday < 0:
        df = df[df['Prev Day Return'] <= prevday]
    if weekday != 'No Weekday Filter':
        df = df[df['Weekday'] == weekday]
    del df['date']
    if consq is None:
        consq = 0
    if consq != 0:
        if consq > 0:
            df = df[df['Consecutive Up/Down Days'] >= consq]
        elif consq < 0:
            df = df[df['Consecutive Up/Down Days'] <= consq]
    if rsi > 0:
        df = df[df['3D RSI'] > rsi]
    if rsi < 0:
        df = df[df['3D RSI'] < -1*rsi]
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
        start_time = date_obj.replace(hour=9, minute=24, second=0).isoformat()
        end_time = (date_obj + pd.offsets.BusinessDay(0)).replace(hour=11, minute=0, second=0).isoformat() 
        blip = np.random.choice([1,2,3])
        if blip == 1:
            zapi = api
        elif blip == 2:
            zapi = api2
        else:
            zapi = api3        
        return zapi.get_bars(ticker, '1Min', start=start_time, end=end_time).df    

    def get_rets(nowdf, timez):
        try:
            first_row_hour = nowdf.index[0].hour
            target_time = pd.to_datetime(f'{first_row_hour}:{timez:02d}:00').time()

            open_time = pd.to_datetime(f'{first_row_hour}:30:00').time()
            row_1430 = nowdf[nowdf.index.time == open_time]
            open = row_1430['open'].iloc[0]
            row_time = nowdf[nowdf.index.time == target_time]
            if target_time == pd.to_datetime(f'{first_row_hour}:24:00').time():
                return round(100 * (open - row_time['open'].iloc[0]) / open, 3)
            else:
                return round(100 * (row_time['open'].iloc[0] - open) / row_time['open'].iloc[0], 3)

        except Exception as e:
            return np.nan
    all_dfs = [get_df(row) for index, row in df.iterrows()]
    # Calculate returns for each dataframe and store them in new columns in df
    df['9:24 to Open'] = [get_rets(df, 24) for df in all_dfs]
    df['1 Min Return'] = [get_rets(df, 31) for df in all_dfs]
    df['3 Min Return'] = [get_rets(df, 33) for df in all_dfs]
    df['5 Min Return'] = [get_rets(df, 35) for df in all_dfs]
    df['10 Min Return'] = [get_rets(df, 40) for df in all_dfs]
    df['15 Min Return'] = [get_rets(df, 45) for df in all_dfs]
    df = df[['Close to Open', '9:24 to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return', 'Open to Close', '# Day High', 'Consecutive Up/Down Days', 'Prev Day Return', '3D RSI']]
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

    return df.dropna(subset = ['9:24 to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return'], how = 'all')


def get_everything2(ticker, amount, weekday = "No Weekday Filter", dailyhigh = 0, consq = 0, offopen = "No Off Open Returns"):
    two = False
    ticker = ticker.replace(" ", "")
    if "/" in ticker:
        two = True
        ticker, back = ticker.split('/', 1)
    else:
        back = ""
    td = pd.to_datetime('today')
    start = td - timedelta(days = 365*8)
    try:
        df = client.get_dataframe(ticker, frequency='Daily', startDate= start, endDate= td - BDay(1))
        df['1 Day Return'] = round(100*(df['adjClose'].shift(-1) - df['adjClose']) / df['adjClose'].shift(-1),3)
        df['3 Day Return'] = round(100*(df['adjClose'].shift(-3) - df['adjClose']) / df['adjClose'].shift(-3),3)
        df['5 Day Return'] = round(100*(df['adjClose'].shift(-5) - df['adjClose']) / df['adjClose'].shift(-5),3)
        df['10 Day Return'] = round(100*(df['adjClose'].shift(-10) - df['adjClose']) / df['adjClose'].shift(-10),3)
        if two:
            one = client.get_dataframe(back, frequency='Daily', startDate= start, endDate= td - BDay(1))
            one['1 Day Return'] = round(100*(one['adjClose'].shift(-1) - one['adjClose']) / one['adjClose'].shift(-1),3)
            one['3 Day Return'] = round(100*(one['adjClose'].shift(-3) - one['adjClose']) / one['adjClose'].shift(-3),3)
            one['5 Day Return'] = round(100*(one['adjClose'].shift(-5) - one['adjClose']) / one['adjClose'].shift(-5),3)
            one['10 Day Return'] = round(100*(one['adjClose'].shift(-10) - one['adjClose']) / one['adjClose'].shift(-10),3)
    except Exception as e:
        print(e)
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
    if two:
        one['Prev Close to Close'] = round(100 * one['adjClose'].pct_change(),3)
        one['Close to Open'] = round(100*(one['adjOpen'].shift(-1) - one['adjClose'])/one['adjClose'],3)
        one['Open to Close'] = round(((one['adjClose'] - one['adjOpen']) / one['adjOpen'] * 100).shift(-1),3)
        for col in ['Prev Close to Close', 'Close to Open', 'Open to Close', '1 Day Return', '3 Day Return', '5 Day Return', '10 Day Return']:
            df[col] = round(df[col] - one[col],3)
    
    if not two:
        df['Price Change'] = df['adjClose'].diff()
        df['Price Change'] = df['Price Change'].apply(lambda x: 1 if x>0 else -1)
        df['Consecutive Up/Down Days'] = df['Price Change'].groupby((df['Price Change'] != df['Price Change'].shift()).cumsum()).cumsum()
    if two:
        df['Price Change'] = np.sign(df['Prev Close to Close'])
        df['Price Change'] = df['Price Change'].apply(lambda x: 1 if x>0 else -1)
        df['Consecutive Up/Down Days'] = df['Price Change'].groupby((df['Price Change'] != df['Price Change'].shift()).cumsum()).cumsum()
    if consq is None:
        consq = 0
    if consq != 0:
        if consq > 0:
            df = df[df['Consecutive Up/Down Days'] >= consq]
        elif consq < 0:
            df = df[df['Consecutive Up/Down Days'] <= consq]

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

        
    def get_df(row, ticker):
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
            blip = np.random.choice([1,2,3])
            if blip == 1:
                zapi = api
            elif blip == 2:
                zapi = api2
            else:
                zapi = api3     
            return zapi.get_bars(ticker, '1Min', start=start_time, end=end_time).df     
        except:
            return pd.DataFrame()

    def get_rets(nowdf, min):
        try:
            if len(nowdf) == 0:
                return np.nan
            open = float(nowdf['open'][0])
            return round(100*(float(nowdf.head(min+1)['open'][-1]) - open)/open,3)
        except:
            return np.nan

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
        df['Prev Day Earnings'] = "Error getting earnings"
    if offopen != 'No Off Open Returns':
        all_dfs = [get_df(row, ticker) for index, row in df.iterrows()]
        df['1 Min Return'] = [get_rets(df, 1) for df in all_dfs]
        df['3 Min Return'] = [get_rets(df, 3) for df in all_dfs]
        df['5 Min Return'] = [get_rets(df, 5) for df in all_dfs]
        df['10 Min Return'] = [get_rets(df, 10) for df in all_dfs]
        df['15 Min Return'] = [get_rets(df, 15) for df in all_dfs]
        df = df[['date', 'Prev Close to Close', 'Close to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return', 'Open to Close', '# Day High', 'All Time High', 'Consecutive Up/Down Days','Prev Day Earnings', 'Weekday']]
        df = df.iloc[::-1]
        if two:
            one.index = one.index.strftime("%Y-%m-%d")
            one = one[one.index.isin(df['date'])].reset_index()
            all_dfs = [get_df(row, back) for index, row in one.iterrows()]
            one['1 Min Return'] = [get_rets(df, 1) for df in all_dfs]
            one['3 Min Return'] = [get_rets(df, 3) for df in all_dfs]
            one['5 Min Return'] = [get_rets(df, 5) for df in all_dfs]
            one['10 Min Return'] = [get_rets(df, 10) for df in all_dfs]
            one['15 Min Return'] = [get_rets(df, 15) for df in all_dfs]
            one = one[['date', 'Prev Close to Close', 'Close to Open', '1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return', 'Open to Close']]
            for col in ['1 Min Return', '3 Min Return', '5 Min Return', '10 Min Return', '15 Min Return']:
                df[col] = round(df[col] - one[col],3)
        return df.dropna().reset_index(drop = True)

    else:
        df = df.iloc[::-1]
        return df[['date', 'Prev Close to Close', 'Close to Open', 'Open to Close', '1 Day Return', '3 Day Return', '5 Day Return', '10 Day Return', '# Day High', 'All Time High' ,'Consecutive Up/Down Days','Prev Day Earnings', 'Weekday']]

import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
from pandas import Timestamp
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
    dcc.Tabs([
        dcc.Tab(label = 'ADR, Off-Open, Intraday Lookback', children = [
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
            html.Div([
                dcc.Input(id='input-ticker', type='text', placeholder='Enter ticker, e.g., GME', debounce=True, style={'margin': '10px', 'width': '29.6%'}),
                dcc.Input(id='input-amount', type='number', placeholder='Enter percent gap up, e.g., 50', debounce=True, style={'margin': '10px', 'width': '29.6%'}),
            ]),
            
            # Row 2: Consecutive + Daily High
            html.Div([
                dcc.Input(id='input-ud', placeholder='Consecutive Up/Down Filter (0 For Default)', type='number', value=None, debounce=True, style={'margin': '10px', 'width': '29.6%'}),
                dcc.Input(id='input-high1', placeholder='Daily High Filter (0 For Default)', type='number', value=None, debounce=True, style={'margin': '10px', 'width': '29.6%'}),
                dcc.Input(id='input-rs1', placeholder='3 Day RSI (0 For Default)', type='number', value=None, debounce=True, style={'margin': '10px', 'width': '29.6%'}),
            ]),
            html.Br(),
            dcc.Dropdown(id='weekday2-filter-dropdown', options=[
            {'label': 'No Weekday Filter', 'value': 'No Weekday Filter'},
            {'label': 'Monday', 'value': 'Monday'},
            {'label': 'Tuesday', 'value': 'Tuesday'},
            {'label': 'Wednesday', 'value': 'Wednesday'},
            {'label': 'Thursday', 'value': 'Thursday'},
            {'label': 'Friday', 'value': 'Friday'},
                ], value = 'No Weekday Filter', placeholder='Select Weekday Filter', style={'margin': '10px', 'width': '50%'}),
            dcc.Input(id='input-pd', placeholder='Prev Day Return (0 For Default)', type='number', value=None, debounce=True, style={'margin': '10px', 'width': '29.6%'}),
            dbc.Button('Submit', id='submit-button', color='primary', n_clicks=0),
            dbc.Button("Download as Excel", id="download-button", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
            html.Div(id='output-table', style={'margin-top': '20px'}),
            dcc.Download(id="download-dataframe2-xlsx"),
            html.Hr(),
            html.H2("Overnight Return - Previous Close-Close Move"),
            html.P("Visualize returns off the open from when a stock has a significant close-close move the previous day:"),
            html.P("You can also visualize returns relative to another stock. I.e. entering QQQ/IWM will display QQQ returns relative to IWM."),
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
            dcc.Input(id='input-updown', placeholder='Consecutive Up/Down Filter Filter (0 For Default)', type='number', value = None, style={'margin': '10px', 'width': '29.6%'}),
            dcc.Dropdown(id='offopen-dropdown', options=[
            {'label': 'No Off Open Returns', 'value': 'No Off Open Returns'},
            {'label': 'Yes Off Open Returns', 'value': 'Yes Off-Open Returns'},
                ], value = 'No Off Open Returns', placeholder='View Off-Open Returns', style={'margin': '10px', 'width': '50%'}),
            dbc.Button('Submit', id='submit-buttons', color='primary', n_clicks=0),
            dbc.Button("Download as Excel", id="download-button2", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
            html.Div(id='doutput-table', style={'margin-top': '20px'}),
            dcc.Download(id="download-dataframe3-xlsx"),
            html.Hr(),
            html.H2("Intraday Return Visualizer"),
            html.P("Visualize returns to the close when a stock makes a specific intraday move."), 
            html.Div([
                html.P([
                    dcc.Input(id='intraday-ticker', type='text', placeholder='Enter ticker, e.g., AAPL', style={'width': '15%', 'display': 'inline-block', 'margin-right': '5px'}),
                    html.Span(" has moved ", style={'margin-right': '5px'}),
                    dcc.Input(id='intraday-amount', type='number', placeholder='Percent Move', style={'width': '12%', 'display': 'inline-block', 'margin-right': '5px'}),
                    html.Span(" % from the ", style={'margin-right': '5px'}),
                    dcc.Dropdown(id='intraday-froms', options=[
                        {'label': 'Open', 'value': 'open'},
                        {'label': 'Previous Close', 'value': 'close'}
                    ], placeholder='Select Open or Close', style={'width': '41%', 'display': 'inline-block', 'margin-right': '5px'})
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'nowrap'}),
                html.P([
                    html.Span(" at ", style={'margin-right': '5px'}),
                    dcc.Dropdown(id='intraday-times', options=[
                        {'label': f'{hour:02d}:{minute:02d}', 'value': f'{hour:02d}:{minute:02d}'}
                        for hour in range(9, 17) for minute in range(0, 60, 15)
                        if not (hour == 16 and minute > 0)

                    ], placeholder='Select Time', style={'width': '40%', 'display': 'inline-block', 'margin-right': '5px'}),
                    html.Span(".")
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'nowrap'})
            ]),
            dbc.Button('Submit', id='intraday-submit', color='primary', n_clicks=0, style={'margin-top': '10px'}),
            dbc.Button("Download as Excel", id="download-button3", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
            html.Div(id='intraday-output-table', style={'margin-top': '20px'}),
            dcc.Download(id="download-dataframe4-xlsx"),
            html.Hr(),
            html.H2("Earnings Long-Term Return Visualizer"),
            html.P("Visualize returns up to 3 months after a specific earnings move."), 
            dbc.Input(id='input-tickerss', type='text', placeholder='Enter ticker, e.g., TSLA'),
            dbc.Input(id='input-amountss', type='number', placeholder='Enter close-close earnings move'),
            dbc.Button('Submit', id='submit-buttonss', color='primary', n_clicks=0),
            html.Div(id='earn-output-table', style={'margin-top': '20px'}),
            html.Hr(),
            html.H2("RSI Backtest Visualizer"),
            html.P("Visualize short term and long term moves corresponding to specific RSI levels"),
            html.P("NOTE: Positive RSI input shows days with >= RSI. Negative RSI input shows days with <= RSI"),
            dbc.Input(id = 'input-tickerrsi', type = 'text', placeholder = 'Enter ticker, e.g., AMD'),
            dbc.Input(id = 'input-days', type = 'number', placeholder = 'Enter No. Day RSI used for calculation, e.g., 14'),
            dbc.Input(id = 'input-rsi', type = 'number', placeholder = 'Enter RSI used for filtering. Negative number = upper bound, Positive number = lower bound.'),
            dbc.Button('Submit', id = 'submit-rsi', color = 'primary', n_clicks = 0),
            html.Div(id = 'rsi-output-table', style = {'margin-top':'20px'}),
        ]), 
        dcc.Tab(label='JPM Index Data Filter', children=[
            html.H1("JPM Index Data Filter", style={'text-align': 'center', 'margin-bottom': '20px'}),

            # Dropdown for selecting 'idx_nm', options sorted alphabetically
            dcc.Dropdown(
                id='idx_nm-dropdown',
                options=[{'label': i, 'value': i} for i in sorted(fy['Index Name'].unique())],  # Sorted idx_nm options
                placeholder="Select an Index",
                style={'width': '50%', 'margin-top': '20px'}
            ),

            # Dropdown for selecting 'idx_chg', dynamically populated
            dcc.Dropdown(
                id='idx_chg-dropdown',
                placeholder="Select a Change",
                style={'width': '50%', 'margin-top': '20px'}
            ),

            # Input for net_adv filter
            # Dropdown for Completed/Expected filter
            dcc.Dropdown(
                    id='ecm-dropdown',
                    options=[{'label': i, 'value': i} for i in ['Completed only', 'Upcoming only', 'Both']],
                    placeholder='Completed/Expected Filter',
                    style={'width': '50%', 'margin-top': '20px'}
            ),
            html.Div([
                html.Label('Filter by net_adv:', style={'font-size': '12px', 'margin-right': '10px'}),
                dcc.Input(
                    id='net_adv-input',
                    type='number',
                    placeholder='Default: 0',
                    value=0,
                    style={'width': '100px', 'height': '20px'}
                )
            ], style={'display': 'flex', 'align-items': 'center', 'margin-top': '20px'}),

            # Input for net_val_M filter
            html.Div([
                html.Label('Filter by net_val_M:', style={'font-size': '12px', 'margin-right': '10px'}),
                dcc.Input(
                    id='net_val_M-input',
                    type='number',
                    placeholder='Default: 0',
                    value=0,
                    style={'width': '100px', 'height': '20px'}
                )
            ], style={'display': 'flex', 'align-items': 'center', 'margin-top': '20px'}),


            # Button to submit the filter request
            html.Button('Get Results', id='filter-button', n_clicks=0, style={'margin-top': '20px'}),
            dbc.Button("Download as Excel", id="download-button9", n_clicks=0, style={'margin-left': '20px', 'font-size': '12px', 'padding': '5px 10px'}),
            # Div to show the filtered results
            html.Div(id='filtered-data', style={'margin-top': '20px'}),
            dcc.Download(id="download-dataframe5-xlsx"),

        ])
    ])
])



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
        
        if isinstance(df, tuple):
            df, err_msg = df
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
        t2 = ticker.upper() + " EQUITY"
        amt2 = amt_threshold*0.01
        stdv = stdevs(t2, data, disc_flag, amt2)
        if disc_flag:
            t = 'discount'
        else:
            t = 'premium'
        result_string += f" This {t} is {stdv} standard deviations from average for {ticker.upper()}."
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
    [
        Input('submit-button', 'n_clicks'),
        Input('input-ticker', 'n_submit'),
        Input('input-amount', 'n_submit'),
        Input('input-high1', 'n_submit'),
        Input('input-rs1', 'n_submit'),
        Input('input-ud', 'n_submit'),
        Input('input-pd', 'n_submit'),
    ],
    [
        State('input-ticker', 'value'),
        State('input-amount', 'value'),
        State('input-high1', 'value'),
        State('input-rs1', 'value'),
        State('input-ud', 'value'),
        State('weekday2-filter-dropdown', 'value'),
        State('input-pd', 'value')
    ]
)
def update_output(n_clicks, s1, s2, s3, s4, s5, s6, ticker, amount, high1, rsi, ud, weekday2, pds):
    if (n_clicks > 0 or s1 or s2 or s3 or s4 or s5) and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything(ticker, amount, high1, ud, weekday2, pds, rsi)            
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column in ['date', 'Prev Day Earnings', '# Day High', 'Consecutive Up/Down Days', "Weekday", "Prev Day Return",'3D RSI']:
                    continue
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
                if column in ['# Day High', 'Consecutive Up/Down Days', "Prev Day Return",'3D RSI']:
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
                page_size=5,
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
    [
        State('input-ticker', 'value'),
        State('input-amount', 'value'),
        State('input-high1', 'value'),
        State('input-rs1', 'value'),
        State('input-ud', 'value'),
        State('weekday2-filter-dropdown', 'value'),
        State('input-pd', 'value')
    ])
def generate_excel(n_clicks, ticker, amount, high1, rsi, ud, weekday2, pds):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything(ticker, amount, high1, ud, weekday2, pds, rsi)
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
     State('input-updown', 'value'),
     State('offopen-dropdown', 'value'),]
)
def update_output(n_clicks, ticker, amount, weekday_filter, dailyhigh, consq, offopen):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything2(ticker, amount, weekday_filter, dailyhigh, consq, offopen)            
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column in ['date', 'Weekday', '# Day High', 'All Time High', 'Prev Day Earnings', 'Consecutive Up/Down Days']:
                    continue
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
                if column in ['All Time High', '# Day High', 'Prev Day Earnings', 'Consecutive Up/Down Days']:
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
                page_size=5,
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
    [State('input-tickers', 'value'),
     State('input-amounts', 'value'),
     State('weekday-filter-dropdown', 'value'),
     State('input-high', 'value'),
     State('input-updown', 'value'),
     State('offopen-dropdown', 'value'),]
)
def generate_excel(n_clicks, ticker, amount, weekday_filter, dailyhigh, consq, offopen):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_everything2(ticker, amount, weekday_filter, dailyhigh, consq, offopen)
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
    Output('intraday-output-table', 'children'),
    Input('intraday-submit', 'n_clicks'),
    [State('intraday-ticker', 'value'), 
     State('intraday-amount', 'value'), 
     State('intraday-froms', 'value'), 
     State('intraday-times', 'value')]
)
def update_intraday_output(n_clicks, ticker, amount, froms, times):
    if n_clicks > 0 and ticker and amount is not None and froms and times:
        try:
            df = returns(ticker, amount, times, froms)
            df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%d %I:%M %p") if pd.notnull(x) else None)

            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column == 'timestamp':
                    continue
                # Convert column values to numeric
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
                strin = "{" + column + "}"
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' > 0', 'column_id': column},
                    'backgroundColor': '#228C22', 'color':'white'
                })
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' < 0', 'column_id': column},
                    'backgroundColor': '#FF6666', 'color':'white'
                })
            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=10,
                sort_action = 'native',
                style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditionals  # Apply color coding
            )
        except Exception as e:
            return html.Div(f"Error fetching data: {str(e)}")
    return html.Div("Submit a ticker and amount to see data.")
@app.callback(
    Output('download-dataframe4-xlsx', 'data'),
    [Input('download-button3', 'n_clicks')],
    [State('intraday-ticker', 'value'), 
     State('intraday-amount', 'value'), 
     State('intraday-froms', 'value'), 
     State('intraday-times', 'value')]
)
def generate_excel(n_clicks, ticker, amount, froms, times):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            df = returns(ticker, amount, times, froms)
            df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%d %I:%M %p") if pd.notnull(x) else None)
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
    Output('earn-output-table', 'children'),
    [Input('submit-buttonss', 'n_clicks')],
    [State('input-tickerss', 'value'),],
    [State('input-amountss', 'value')]
)
def update_output(n_clicks, ticker, amount):
    if n_clicks > 0 and ticker and amount is not None:
        try:
            amount = float(amount)
            df = get_earnings(ticker, amount)            
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column in ['date']:
                    continue
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
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
                page_size=5,
                style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditionals  # Apply color coding
            )
        except Exception as e:
            return html.Div(f"Error fetching data: {str(e)}")
    return html.Div("Submit a ticker and amount to see data.")
@app.callback(
    Output('rsi-output-table', 'children'),
    [Input('submit-rsi', 'n_clicks')],
    [State('input-tickerrsi', 'value'),],
    [State('input-days', 'value')],
    [State('input-rsi', 'value')]
)
def update_output(n_clicks, ticker, days, rs):
    if n_clicks > 0 and ticker and days and rs is not None:
        try:
            days = int(days)
            rs = float(rs)
            df = rsi(ticker, days, rs)     
            # Color coding for values in each column
            style_data_conditionals = []
            for column in df.columns:
                if column in ['date', 'RSI', 'Close Price']:
                    continue
                df[column] = pd.to_numeric(df[column], errors='coerce')
            style_data_conditionals = []
            for column in df.columns:
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
                page_size=5,
                style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_data_conditionals  # Apply color coding
            )
        except Exception as e:
            return html.Div(f"Error fetching data: {str(e)}")
    return html.Div("Submit a ticker and amount to see data.")
@app.callback(
    [Output('idx_chg-dropdown', 'options'),
     Output('idx_chg-dropdown', 'value')],  # Also update the 'value' of the dropdown
    Input('idx_nm-dropdown', 'value')
)
def set_idx_chg_options(selected_idx_nm):
    if selected_idx_nm:
        # Filter the DataFrame based on selected 'idx_nm' and return unique 'idx_chg' values
        filtered_values = fy[fy['Index Name'] == selected_idx_nm]['Index Change'].dropna().unique()  # Drop NaN (null) values
        filtered_values = sorted(filtered_values)
        return [{'label': i, 'value': i} for i in filtered_values if i is not None], None  # Ensure no None values
    else:
        return [], None

# Callback to filter the dataframe based on user input from dropdowns and inputs
@app.callback(
    Output('filtered-data', 'children'),
    Input('filter-button', 'n_clicks'),
    State('idx_nm-dropdown', 'value'),
    State('idx_chg-dropdown', 'value'),
    State('net_adv-input', 'value'),
    State('net_val_M-input', 'value'),
    State('ecm-dropdown', 'value')
)

def filter_dataframe(n_clicks, selected_idx_nm, selected_idx_chg, net_adv_value, net_val_M_value, ecm_value):
    if n_clicks > 0:
        noup = True
        if ecm_value == 'Completed only':
            common_rows = fy.merge(fyu, on=list(fy.columns), how='inner')

            # Step 2: Drop the common rows from fy
            fy_filtered = fy[~fy.index.isin(common_rows.index)]
            filtered_df = fy_filtered.copy()
            
        elif ecm_value == 'Upcoming only':
            filtered_df = fyu.copy()
            npup = False
            filtered_df = filtered_df.sort_values(by = 'Effective', ascending = True)
        elif ecm_value == 'Both':
            filtered_df = fy.copy()
        else:
            filtered_df = fy.copy()

        # Filter by 'idx_nm'
        if selected_idx_nm:
            filtered_df = filtered_df[filtered_df['Index Name'] == selected_idx_nm]

        # Filter by 'idx_chg'
        if selected_idx_chg:
            filtered_df = filtered_df[filtered_df['Index Change'] == selected_idx_chg]

        # Filter by net_adv
        if net_adv_value != 0:
            if net_adv_value > 0:
                filtered_df = filtered_df[filtered_df['Net ADV'] >= net_adv_value]
            else:
                filtered_df = filtered_df[filtered_df['Net ADV'] <= net_adv_value]

        # Filter by net_val_M
        if net_val_M_value != 0:
            if net_val_M_value > 0:
                filtered_df = filtered_df[filtered_df['Net Value (mm)'] >= net_val_M_value]
            else:
                filtered_df = filtered_df[filtered_df['Net Value (mm)'] <= net_val_M_value]

        # Format columns to two decimal places
        for col in ['Value (mm)', 'Shares (mm)', 'ADV', 'Net Value (mm)', 'Net Shares (mm)', 'Net ADV']:
            filtered_df[col] = filtered_df[col].round(2)

        # Format 'Effective' and 'Announcement Date' columns
        filtered_df['Effective'] = pd.to_datetime(filtered_df['Effective']).dt.strftime('%Y-%m-%d')
        filtered_df['Announcement Date'] = pd.to_datetime(filtered_df['Announcement Date']).dt.strftime('%Y-%m-%d')
        filtered_df = filtered_df.rename(columns = {'Weight Change': 'Weight Change (%)'})
        filtered_df['Weight Change (%)'] = round(100*filtered_df['Weight Change (%)'], 3)
        cols = filtered_df.columns.tolist()
        cols.insert(cols.index('Effective') + 1, cols.pop(cols.index('Announcement Date')))
        filtered_df = filtered_df[cols]
        # Display the filtered DataFrame as an HTML table
        del filtered_df['Region']
        
        style_data_conditionals = []
        if noup:
            for column in ['Prev Close to Effective Close', 'Effective Open to Close', 'Effective Close to Next Open', 'Effective T+1 Open to Close', 'Annc Close to Eff. Close', 'Annc Close to Next Open', 'Annc T+1 Open to Close']:
                strin = "{" + column + "}"
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' > 0', 'column_id': column},
                    'backgroundColor': '#228C22', 'color':'white'
                })
                style_data_conditionals.append({
                    'if': {'filter_query': strin + ' < 0', 'column_id': column},
                    'backgroundColor': '#FF6666', 'color':'white'
                })

        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values('Effective', ascending=False)
            return dash_table.DataTable(
                data=filtered_df.to_dict('records'),
                sort_action='native',
                columns=[{'name': col, 'id': col} for col in filtered_df.columns],
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',  # Ensuring the table uses the full width available
                    'minWidth': '100%',  # Minimum width of the table to ensure all columns are visible
                },                fixed_columns={'headers': True, 'data': 1},
                style_header={
                    'fontWeight': 'bold',  # Bold header
                    'backgroundColor': '#F9F9F9',  # Light grey header background
                    'textAlign': 'center'  # Center-align header text
                },
                style_data_conditional=style_data_conditionals,
                # Conditionally style headers for specific columns
                style_header_conditional=[
                    {
                        'if': {'column_id': col},
                        'backgroundColor': '#FFFF00'
                    } for col in ['Net Value (mm)', 'Net Shares (mm)', 'Net ADV']
                ],
            )
        else:
            return html.P("No matching data found.")
    return None
@app.callback(
    Output('download-dataframe5-xlsx', 'data'),
    [Input('download-button9', 'n_clicks')],
    [State('idx_nm-dropdown', 'value'),
     State('idx_chg-dropdown', 'value'),
     State('net_adv-input', 'value'),
     State('net_val_M-input', 'value'),
     State('ecm-dropdown', 'value')]
)
def generate_excel(n_clicks, selected_idx_nm, selected_idx_chg, net_adv_value, net_val_M_value, ecm_value):
    if n_clicks > 0:
        # Initialize filtered DataFrame based on ECM value
        if ecm_value == 'Completed only':
            common_rows = fy.merge(fyu, on=list(fy.columns), how='inner')
            fy_filtered = fy[~fy.index.isin(common_rows.index)]
            filtered_df = fy_filtered.copy()
        elif ecm_value == 'Upcoming only':
            filtered_df = fyu.copy()
            filtered_df = filtered_df.sort_values(by = 'Effective', ascending = True)
        elif ecm_value == 'Both':
            filtered_df = fy.copy()
        else:
            filtered_df = fy.copy()

        if selected_idx_nm:
            filtered_df = filtered_df[filtered_df['Index Name'] == selected_idx_nm]

        # Filter by 'idx_chg'
        if selected_idx_chg:
            filtered_df = filtered_df[filtered_df['Index Change'] == selected_idx_chg]

        # Filter by net_adv
        if net_adv_value != 0:
            if net_adv_value > 0:
                filtered_df = filtered_df[filtered_df['Net ADV'] >= net_adv_value]
            else:
                filtered_df = filtered_df[filtered_df['Net ADV'] <= net_adv_value]

        # Filter by net_val_M
        if net_val_M_value != 0:
            if net_val_M_value > 0:
                filtered_df = filtered_df[filtered_df['Net Value (mm)'] >= net_val_M_value]
            else:
                filtered_df = filtered_df[filtered_df['Net Value (mm)'] <= net_val_M_value]

        # Format columns to two decimal places
        for col in ['Value (mm)', 'Shares (mm)', 'ADV', 'Net Value (mm)', 'Net Shares (mm)', 'Net ADV']:
            filtered_df[col] = filtered_df[col].round(2)

        # Format 'Effective' and 'Announcement Date' columns
        filtered_df['Effective'] = pd.to_datetime(filtered_df['Effective']).dt.strftime('%Y-%m-%d')
        filtered_df['Announcement Date'] = pd.to_datetime(filtered_df['Announcement Date']).dt.strftime('%Y-%m-%d')
        filtered_df = filtered_df.rename(columns = {'Weight Change': 'Weight Change (%)'})
        filtered_df['Weight Change (%)'] = round(100*filtered_df['Weight Change (%)'], 3)
        cols = filtered_df.columns.tolist()
        cols.insert(cols.index('Effective') + 1, cols.pop(cols.index('Announcement Date')))
        filtered_df = filtered_df[cols]
        # Export to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        xlsx_data = output.getvalue()

        return dcc.send_bytes(xlsx_data, "output.xlsx")
    return None


if __name__ == '__main__':
    app.run_server(debug=True, port = 8051)
