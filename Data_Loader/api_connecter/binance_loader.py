import pandas as pd
import numpy as np
import os, sys
import requests
import time

PROJECT_ROOT = '..'
sys.path.extend([PROJECT_ROOT, '../..']) 
from prepare_data import ExchangeData
from message_manager import MessageManager
message = MessageManager()

class BinanceLoader(ExchangeData):

    def __init__(self, exchange, symbol_type, long_short_ratio_type='global', timezone='Asia/Taipei'):
        super().__init__(exchange, symbol_type, long_short_ratio_type)
        self.timezone = timezone


    def get_ts13(self, datetime:str,):
        ts13 = int(pd.to_datetime(datetime).tz_localize(self.timezone).timestamp()*1000)
        return ts13
  

    def fetch_ohlcv(
            self, 
            url, 
            symbol, 
            freq, 
            start_ts13,
            limit, 
            columns, 
            need_col = ["datetime", "open", "high", "low", "close", "quote_asset_volume"],
            end_ts13 = None,
        ):

        """
        Mind
        - timezone = 'UTC'
        - spot: see https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        - usd: see https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        - coin: see https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
        """
        item = 'ohlcv'
        data_li = []
        params = {
            'symbol': symbol, 
            'interval': freq, 
            'limit':limit,
            'endTime': end_ts13,
        }
        while True:
            message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
            # timezone of binance is UTC+0
            params['startTime'] = start_ts13  # update start_time
            data = requests.get(url, params=params).json()
            data_li.extend(data) 
            if len(data) < limit:
                break
            last = -1    
            start_ts13 = int(data[last][0]) + 1

        # form dataframe
        if data_li: 
            try:
                df = pd.DataFrame(data_li)
                df.columns = columns
                df = df[need_col].set_index(['datetime']).astype('float64').sort_index()
            except ValueError:
                df = pd.DataFrame(data_li)
                df.columns = columns
                df = df[need_col].set_index(['datetime']).astype('float64').sort_index()
                df = df.apply(pd.to_numeric, errors='coerce').sort_index()

            df.index = pd.to_datetime(df.index.astype(float), unit='ms').tz_localize('UTC').tz_convert(self.timezone)
            return df
        else: 
            return pd.DataFrame()


    def fetch_funding_rate(
            self, 
            url, 
            symbol, 
            start_ts13,
            limit, 
            columns, 
            need_col = ["datetime", "funding_rate"],
            end_ts13 = None,
        ):

        """
        Mind
        - timezone = 'UTC'
        - usd: see https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
        - coin: see https://binance-docs.github.io/apidocs/delivery/en/#get-funding-rate-history-of-perpetual-futures
        """
        item = 'funding_rate'
        data_li = []
        params = {
            'symbol': symbol, 
            'limit':limit,
            'endTime': end_ts13,
        }
        while True:
            message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
            # timezone of binance is UTC+0
            params['startTime'] = start_ts13                        # update start_time
            data = requests.get(url, params=params).json()
            data_li.extend(data) 
            if len(data) < limit:
                break
            last = -1    
            start_ts13 = int(data[last]['fundingTime']) + 1


        # form dataframe
        if data_li: 
            df = pd.DataFrame(data_li)
            df.columns = columns
            df = df[need_col].set_index(['datetime']).astype('float64').sort_index()
            df.index = pd.to_datetime(df.index.astype(float), unit='ms').tz_localize('UTC').tz_convert(self.timezone)
            return df
        else: 
            return pd.DataFrame()


    def fetch_long_short_ratio(
            self, 
            url, 
            symbol, 
            freq, 
            start_ts13,
            limit, 
            columns, 
            need_col = ["datetime", "long_short_ratio", "long_account", "short_account"],
            end_ts13 = None,
        ):

        """
        Mind
        - timezone = 'UTC'
        - usd: see https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-accounts
        """
        item = 'long_short_ratio'
        data_li = []
        params = {
            'symbol': symbol, 
            'period': freq, 
            'limit':limit,
            'endTime': end_ts13,
        }
        while True:
            message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
            # timezone of binance is UTC+0
            params['startTime'] = start_ts13  # update start_time
            data = requests.get(url, params=params).json()
            data_li.extend(data) 
            if len(data) < limit:
                break
            last = -1    
            start_ts13 = int(data[last]['timestamp']) + 1

        # form dataframe
        if data_li: 
            try:
                df = pd.DataFrame(data_li)
                df.columns = columns
                df = df[need_col].set_index(['datetime']).astype('float64').sort_index()
            except ValueError:
                df = pd.DataFrame(data_li)
                df.columns = columns
                df = df[need_col].set_index(['datetime']).astype('float64').sort_index()
                df = df.apply(pd.to_numeric, errors='coerce').sort_index()

            df.index = pd.to_datetime(df.index.astype(float), unit='ms').tz_localize('UTC').tz_convert(self.timezone)
            return df
        else: 
            return pd.DataFrame()


    async def update_ohlcv(self, start:str, freq:str, symbol_li:list=None, end:str=None):
        """
        Update ohlcv for the given symbols.

        Args:
        - start (str): Start datetime. (e.g., '2023-1-1 12:00:00+08:00')
        - freq (str): Frequency of data (e.g., '30m', '1d', '1h'). 
        - symbol_li (list): List of symbols to download.
        - end (str): End datetime. (see start)

        Example:
            update_ohlcv('2021-1-1', '30m', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No symbol checking in symbol_li.
        - timezone = 'UTC' in binance.
        - update the data at the time of execution if no 'end'. 
            - ( all symbols with the same ending datetime )
        """
        item = 'ohlcv'
        save_dir = os.path.join(PROJECT_ROOT, 'data_base', self.exchange, self.symbol_type, item, freq)
        os.makedirs(save_dir, exist_ok=True)

        url = self.get_end_point() + self.get_suffix_kline()
        start_ts13 = self.get_ts13(start)
        end_ts13 = self.get_ts13(end) if end else int(time.time()*1000)
        limit = self.get_limit_kline()
        columns = self.get_columns_kline()

        if not symbol_li:
            url_exchange_info = self.get_end_point() + self.get_suffix_exchange_info()
            exchange_info = requests.get(url_exchange_info).json()
            symbol_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

        for symbol in symbol_li:
            symbol_amount = len(symbol_li)
            message.updating(item, symbol_li.index(symbol)+1, symbol_amount)

            file_name = f"{symbol}_ohlcv.pkl"
            file_path = os.path.join(save_dir, file_name)
            df_ori = pd.read_pickle(file_path) if os.path.isfile(file_path) else pd.DataFrame()

            # no file or empty
            if not os.path.isfile(file_path) or df_ori.empty:
                message.necessary_creating(item, symbol)
                df_updated = self.fetch_ohlcv(
                    url = url,
                    symbol = symbol,
                    freq = freq,
                    start_ts13 = start_ts13,
                    end_ts13 = end_ts13,
                    limit = limit,
                    columns = columns,
                )
                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.donload_completed(item, symbol, file_path)

            # file exists and not empty
            else:          
                df_ori = pd.read_pickle(file_path)
                start_ts13_ori = int(df_ori.index[0].timestamp()*1000)
                end_ts13_ori = int(df_ori.index[-1].timestamp()*1000)
                update_li = [df_ori]

                if (start_ts13 > start_ts13_ori) & (end_ts13 < end_ts13_ori):
                    message.unnecessary_update(item, symbol)
                    continue
                
                # head
                if start_ts13 < start_ts13_ori:
                    df_new_head = self.fetch_ohlcv(
                        url = url,
                        symbol = symbol,
                        start_ts13 = start_ts13,
                        end_ts13 = start_ts13_ori-1,
                        freq = freq,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.insert(0, df_new_head) 

                # tail
                if end_ts13 > end_ts13_ori:
                    df_new_tail = self.fetch_ohlcv(
                        url = url,
                        symbol = symbol,
                        start_ts13 = end_ts13_ori+1,
                        end_ts13 = end_ts13,
                        freq = freq,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.append(df_new_tail)

                df_updated = pd.concat(update_li)
                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.update_completed(item, symbol)


    async def update_funding_rate(self, start:str, symbol_li:list=None, end:str=None):
        """
        Update funding rate for the given symbols.

        Args:
        - start (str): Start datetime. (e.g., '2023-1-1 12:00:00+08:00')
        - symbol_li (list): List of symbols to download.
        - end (str): End datetime. (see start)

        Example:
            update_ohlcv('2021-1-1', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No symbol checking in symbol_li.
        - timezone = 'UTC' in binance.
        - update the data at the time of execution if no 'end'. 
            - ( all symbols with the same ending datetime )
        """
        item = 'funding_rate'
        save_dir = os.path.join(PROJECT_ROOT, 'data_base', self.exchange, self.symbol_type, item)
        os.makedirs(save_dir, exist_ok=True)

        url = self.get_end_point() + self.get_suffix_funding_rate()
        start_ts13 = self.get_ts13(start)
        end_ts13 = self.get_ts13(end) if end else int(time.time()*1000)
        limit = self.get_limit_funding_rate()
        columns = self.get_columns_funding_rate()

        if not symbol_li:
            url_exchange_info = self.get_end_point() + self.get_suffix_exchange_info()
            exchange_info = requests.get(url_exchange_info).json()
            symbol_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

        for symbol in symbol_li:
            symbol_amount = len(symbol_li)
            message.updating(item, symbol_li.index(symbol)+1, symbol_amount)

            file_name = f"{symbol}_funding_rate.pkl"
            file_path = os.path.join(save_dir, file_name)
            df_ori = pd.read_pickle(file_path) if os.path.isfile(file_path) else pd.DataFrame()

            # no file or empty
            if not os.path.isfile(file_path) or df_ori.empty:
                message.necessary_creating(item, symbol)
                df_updated = self.fetch_funding_rate(
                    url = url,
                    symbol = symbol,
                    start_ts13 = start_ts13,
                    end_ts13 = end_ts13,
                    limit = limit,
                    columns = columns,
                )
                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.donload_completed(item, symbol, file_path)

            # file exists and not empty
            else:          
                df_ori = pd.read_pickle(file_path)
                start_ts13_ori = int(df_ori.index[0].timestamp()*1000)
                end_ts13_ori = int(df_ori.index[-1].timestamp()*1000)
                update_li = [df_ori]

                if (start_ts13 > start_ts13_ori) & (end_ts13 < end_ts13_ori):
                    message.unnecessary_update(item, symbol)
                    continue
                
                # head
                if start_ts13 < start_ts13_ori:
                    df_new_head = self.fetch_funding_rate(
                        url = url,
                        symbol = symbol,
                        start_ts13 = start_ts13,
                        end_ts13 = start_ts13_ori-1,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.insert(0, df_new_head) 

                # tail
                if end_ts13 > end_ts13_ori:
                    df_new_tail = self.fetch_funding_rate(
                        url = url,
                        symbol = symbol,
                        start_ts13 = end_ts13_ori+1,
                        end_ts13 = end_ts13,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.append(df_new_tail)

                df_updated = pd.concat(update_li)
                df_updated['funding_rate'] =pd.to_numeric(df_updated['funding_rate'], errors='coerce').astype('float64')

                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.update_completed(item, symbol)


    async def update_long_short_ratio(self, start:str, freq:str, symbol_li:list=None, end:str=None,):
        """
        Update long_short_ratio for the given symbols.

        Args:
        - start (str): Start datetime. (e.g., '2023-1-1 12:00:00+08:00')
        - freq (str): Frequency of data (e.g., '30m', '1d', '1h'). 
        - symbol_li (list): List of symbols to download.
        - end (str): End datetime. (see start)

        Example:
            update_ohlcv('2021-1-1', '30m', ['BTCUSDT', 'ETHUSDT'])

        Mind
        - No symbol checking in symbol_li.
        - timezone = 'UTC' in binance.
        - update the data at the time of execution if no 'end'. 
            - ( all symbols with the same ending datetime )
        """
        item = 'long_short_ratio'
        save_dir = os.path.join(PROJECT_ROOT, 'data_base', self.exchange, self.symbol_type, item, freq)
        os.makedirs(save_dir, exist_ok=True)

        url = self.get_end_point() + self.get_suffix_long_short_ratio()
        start_ts13 = self.get_ts13(start)
        end_ts13 = self.get_ts13(end) if end else int(time.time()*1000)
        limit = self.get_limit_long_short_ratio()
        columns = self.get_columns_long_short_ratio()

        if not symbol_li:
            url_exchange_info = self.get_end_point() + self.get_suffix_exchange_info()
            exchange_info = requests.get(url_exchange_info).json()
            symbol_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

        for symbol in symbol_li:
            symbol_amount = len(symbol_li)
            message.updating(item, symbol_li.index(symbol)+1, symbol_amount)

            file_name = f"{symbol}_{self.long_short_ratio_type}.pkl"
            file_path = os.path.join(save_dir, file_name)
            df_ori = pd.read_pickle(file_path) if os.path.isfile(file_path) else pd.DataFrame()

            # no file or empty
            if not os.path.isfile(file_path) or df_ori.empty:
                message.necessary_creating(item, symbol)
                df_updated = self.fetch_long_short_ratio(
                    url = url,
                    symbol = symbol,
                    freq = freq,
                    start_ts13 = start_ts13,
                    end_ts13 = end_ts13,
                    limit = limit,
                    columns = columns,
                )
                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.donload_completed(item, symbol, file_path)

            # file exists and not empty
            else:          
                df_ori = pd.read_pickle(file_path)
                start_ts13_ori = int(df_ori.index[0].timestamp()*1000)
                end_ts13_ori = int(df_ori.index[-1].timestamp()*1000)
                update_li = [df_ori]

                if (start_ts13 > start_ts13_ori) & (end_ts13 < end_ts13_ori):
                    message.unnecessary_update(item, symbol)
                    continue
                
                # head
                if start_ts13 < start_ts13_ori:
                    df_new_head = self.fetch_long_short_ratio(
                        url = url,
                        symbol = symbol,
                        start_ts13 = start_ts13,
                        end_ts13 = start_ts13_ori-1,
                        freq = freq,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.insert(0, df_new_head) 

                # tail
                if end_ts13 > end_ts13_ori:
                    df_new_tail = self.fetch_long_short_ratio(
                        url = url,
                        symbol = symbol,
                        start_ts13 = end_ts13_ori+1,
                        end_ts13 = end_ts13,
                        freq = freq,
                        limit = limit,
                        columns = columns,
                    )
                    update_li.append(df_new_tail)

                df_updated = pd.concat(update_li)
                if df_updated.empty:
                    message.empty_warning(item, symbol) 
                else:
                    df_updated.to_pickle(file_path)
                    message.update_completed(item, symbol)