import pandas as pd
import numpy as np
import requests
import os, sys
import ccxt

class BinanceLoader():

    def __init__(self, timezone='Asia/Taipei'):
        self.spot_client = ccxt.binance()
        self.timezone = timezone
        self.end_point = "https://api.binance.com"

    def download_spot_ohlcv(self, start:str, freq:str, spot_li:list=None, limit:int=1000):
        """
        Download spot OHLCV data for the given symbols.

        Args:
        - start (str): Start date. (e.g., '2023-1-1 12:00:00+08:00')
        - freq (str): Frequency of data (e.g., '30m', '1d', '1h'). 
        - spot_li (list): List of symbols to download.
        - limit (int): Limit of data points to fetch.

        Example:
            download_spot_ohlcv('2023-1-1 12:00:00+08:00', '30m', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No check if_spot in spot_li.
        - max_limit = 1000 for spot markets
        - timezone = 'UTC'
        """
        save_dir = os.path.join(os.path.dirname(__file__), 'data_base', 'binance', 'spot', freq)
        os.makedirs(save_dir, exist_ok=True)

        if not spot_li:
            suffix = '/api/v3/exchangeInfo'
            url = self.end_point + suffix

            response = requests.get(url)
            exchange_info = response.json()
            
            spot_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

        for symbol in spot_li:
            symbol_amount = len(spot_li)
            print(f"downloading {spot_li.index(symbol)+1}/{symbol_amount}")

            df = self.fetch_spot_ohlcv(symbol, start, freq, limit)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(self.timezone)
            df = df.set_index(['datetime', 'code']).sort_index()

            file_name = f"{symbol}_ohlcv.pkl"
            file_path = os.path.join(save_dir, file_name)
            df.to_pickle(file_path)

            print(f"{symbol} has been downloaded to {file_path}")


    def fetch_spot_ohlcv(self, symbol, start, freq, limit=1000):
        """
        Mind
        - max_limit = 1000 for spot markets
        - timezone = 'UTC'
        """
        ohlcv_list = []
        start_ts13 = self.spot_client.parse8601(start)
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        while True:
            ohlcv = self.spot_client.fetch_ohlcv(symbol, freq, since=start_ts13, limit=limit)
            ohlcv_list.extend(ohlcv)
            print(f"fetching {symbol} from {pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC')}")
            if len(ohlcv) < limit:
                break    
            start_ts13 = ohlcv[-1][0]

        df = pd.DataFrame(ohlcv_list, columns=columns)
        df['code'] = symbol

        return df