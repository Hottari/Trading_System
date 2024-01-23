import pandas as pd
import numpy as np
import os, sys
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, PROJECT_ROOT) 
from prepare_data import ExchangeData

class BinanceLoader(ExchangeData):

    def __init__(self, exchange, symbol_type, timezone='Asia/Taipei'):
        super().__init__(exchange, symbol_type)
        self.timezone = timezone


    def get_ts13(self, datetime:str,):
        ts13 = int(pd.to_datetime(datetime).tz_localize(self.timezone).timestamp()*1000)
        return ts13
  
    
    def fetch_ohlcv(self, url, symbol, freq, start_ts13, limit, columns, need_col=["datetime", "open", "high", "low", "close", "volume"]):
        """
        Mind
        - timezone = 'UTC'
        - spot: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        - usd: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        - coin: https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
        """
        ohlcv_list = []
        params = {
            'symbol': symbol, 
            'interval': freq, 
            'limit':limit,
        }
        while True:
            print(f"fetching {symbol} from {pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC')}") # timezone of binance is UTC+0
            params['startTime'] = start_ts13                                                            # update start_time
            ohlcv = requests.get(url, params=params).json()
            ohlcv_list.extend(ohlcv) 
            if len(ohlcv) < limit:
                break    
            start_ts13 = ohlcv[-1][0] + 1

        df = pd.DataFrame(ohlcv_list, columns=columns)[need_col].set_index(['datetime']).sort_index()
        df.index = pd.to_datetime(df.index, unit='ms').tz_localize('UTC').tz_convert(self.timezone)
        return df
    

    def download_ohlcv(self, start:str, freq:str, symbol_li:list=None,):
        """
        Download ohlcv for the given symbols.

        Args:
        - start (str): Start date. (e.g., '2023-1-1 12:00:00+08:00')
        - freq (str): Frequency of data (e.g., '30m', '1d', '1h'). 
        - symbol_li (list): List of symbols to download.

        Example:
            download_ohlcv('2023-1-1 12:00:00+08:00', '30m', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No symbol checking in symbol_li.
        - timezone = 'UTC' in binance
        """
        save_dir = os.path.join(PROJECT_ROOT, 'data_base', self.exchange, self.symbol_type, freq)
        os.makedirs(save_dir, exist_ok=True)

        url = self.get_end_point() + self.get_suffix_kline()
        start_ts13 = self.get_ts13(start)
        limit = self.get_limit_kline()
        columns = self.get_columns_kline()
        
        if not symbol_li:
            url_exchange_info = self.get_end_point() + self.get_suffix_exchange_info()
            exchange_info = requests.get(url_exchange_info).json()
            symbol_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

        for symbol in symbol_li:
            symbol_amount = len(symbol_li)
            print(f"downloading {symbol_li.index(symbol)+1}/{symbol_amount} symbol")

            df = self.fetch_ohlcv(
                url = url,
                symbol = symbol,
                start_ts13 = start_ts13,
                freq = freq,
                limit = limit,
                columns = columns,
            )
            
            file_name = f"{symbol}_ohlcv.pkl"
            file_path = os.path.join(save_dir, file_name)
            df.to_pickle(file_path)
            print(f"{symbol} has been downloaded to {file_path}")


    def update_ohlcv(self, freq:str, symbol_li:list=None,):
        """
        Update ohlcv for the given symbols.

        Args:
        - freq (str): Frequency of data (e.g., '30m', '1d', '1h'). 
        - symbol_li (list): List of symbols to download.

        Example:
            download_ohlcv('30m', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No symbol checking in symbol_li.
        - timezone = 'UTC' in binance
        """
        save_dir = os.path.join(PROJECT_ROOT, 'data_base', self.exchange, self.symbol_type, freq)
        os.makedirs(save_dir, exist_ok=True)

        url = self.get_end_point() + self.get_suffix_kline()
        limit = self.get_limit_kline()
        columns = self.get_columns_kline()
        
        if not symbol_li:
            file_li = os.listdir(save_dir)
            symbol_li = [i.split('_')[0] for i in file_li]

        for symbol in symbol_li:
            symbol_amount = len(symbol_li)
            print(f"updating {symbol_li.index(symbol)+1}/{symbol_amount} symbol")

            file_name = f"{symbol}_ohlcv.pkl"
            file_path = os.path.join(save_dir, file_name)
            
            df_ori = pd.read_pickle(file_path)
            start_ts13 = int(df_ori.index[-1].timestamp()*1000) + 1
            df_new = self.fetch_ohlcv(
                url = url,
                symbol = symbol,
                start_ts13 = start_ts13,
                freq = freq,
                limit = limit,
                columns = columns,
            )

            df_updated = pd.concat([df_ori, df_new])
            df_updated.to_pickle(file_path)
            print(f"{symbol} has been updated")

