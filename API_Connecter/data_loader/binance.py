import pandas as pd
import numpy as np
import os, sys
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, PROJECT_ROOT) 
from prepare_data import ExchangeData

class BinanceLoader(ExchangeData):

    def __init__(self, exchange, symbol_type):
        super().__init__(exchange, symbol_type)


    def get_ts13(self, datetime:str, timezone:str='Asia/Taipei'):
        ts13 = int(pd.to_datetime(datetime).tz_localize(timezone).timestamp()*1000)
        return ts13
  
    
    def fetch_ohlcv(self, url, symbol, start_ts13, freq, limit, columns, need_col=["datetime", "open", "high", "low", "close", "volume"]):
        """
        Mind
        - timezone = 'UTC'
        - spot: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        - usd: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        - coin: https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
        """

        params = {
            'symbol': symbol, 
            'interval': freq, 
            'startTime': start_ts13, 
            'limit':limit,
        }

        data = requests.get(url, params=params).json()
        df = pd.DataFrame(data, columns=columns)
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        
        return df[need_col].set_index(['datetime'])

