import pandas as pd
import numpy as np
import requests

import os, sys
import ccxt


"""
max_limit = 1000 for spot markets
"timezone":"UTC"

"""
class BinanceLoader():

    def __init__(self):
        pass


def download_spto_ohlcv(freq, spot_li=None):
    if not spot_li:
        api_url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(api_url)
        exchange_info = response.json()
        spot_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']



def fetch_ohlcv_data(exchange, symbol, start, freq, limit=1000):
    ohlcv_list = []
    from_ts = exchange.parse8601(start)

    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, freq, since=from_ts, limit=limit)
        ohlcv_list.extend(ohlcv)
        
        if len(ohlcv) < limit:
            break
        
        from_ts = ohlcv[-1][0]

    df = pd.DataFrame(ohlcv_list, columns=columns)
    df['code'] = symbol
    #df['date'] = pd.to_datetime(df['date'], unit='ms').dt.tz_localize('UTC')

    return df

def fetch_all_ohlcv_data(exchange, symbol_list, freq, start):
    df_li = [fetch_ohlcv_data(exchange, symbol, start, freq).set_index(['date', 'code']) for symbol in symbol_list]
    df_all = pd.concat(df_li, axis='rows').reset_index()
    df_all['date'] = pd.to_datetime(df_all['date'], unit='ms').dt.tz_localize('UTC')
    df_all = df_all.set_index(['date', 'code']).sort_index()
    
    return df_all

    