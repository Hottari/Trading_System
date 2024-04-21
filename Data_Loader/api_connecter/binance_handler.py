import pandas as pd
import numpy as np
import os, sys
import aiohttp

from binance.um_futures import UMFutures
from binance.error import ClientError
from datetime import datetime

PROJECT_ROOT = '..'
sys.path.extend([PROJECT_ROOT, '../..']) 
from message_manager import MessageManager
message = MessageManager()

class BinanceHandler():

    def __init__(self,):
        pass

    def form_dataframe(self, columns, need_col, data_li=[]):
        if data_li: 
            df = pd.DataFrame(data_li)
            df.columns = columns
            df = df[need_col].set_index(['datetime'])
            df.index = pd.to_datetime(df.index.astype(float), unit='ms').tz_localize('UTC')
            return df.sort_index()
        else: 
            return pd.DataFrame()


    def fetch_all_trades(
            self, 
            symbol, api_key, api_secret, 
            start_ts13, end_ts13=pd.to_datetime(datetime.now()).value // 10**6, 
            interval_limit = 60*60*24*7 * 1000,             # binance 7 days limit to fetch trades
            limit = 1000, recvWindow = 1000000,             # https://binance-docs.github.io/apidocs/futures/en/#signed-trade-and-user_data-endpoint-security
            )->list:
        all_trades = []
        while start_ts13 < end_ts13:
            client = UMFutures(
                key = api_key,
                secret = api_secret,
            )
            try:
                response = client.get_account_trades(
                    limit = limit, recvWindow = recvWindow,
                    symbol = symbol, 
                    startTime = start_ts13,
                )
                all_trades.extend(response)
            except ClientError as error:
                print(f"Error status:{error.status_code}, code:{error.error_code}, message:{error.error_message}")
            start_ts13 = min(start_ts13+interval_limit, end_ts13) + 1
        return all_trades



    async def fetch_ohlcv(
            self, 
            url, symbol, freq, limit, columns,
            start_ts13,
            end_ts13 = None, 
            need_col = ["datetime", "open", "high", "low", "close", "quote_asset_volume"],
            **kwargs
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
        params = {k: v for k, v in params.items() if v is not None}     # none in params will make sesstion.get() error
        async with aiohttp.ClientSession() as session:
            while True:
                message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
                params['startTime'] = start_ts13                        # update start_time
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                data_li.extend(data) 
                if len(data) < limit:
                    break
                last = -1    
                start_ts13 = int(data[last][0]) + 1
        df = self.form_dataframe(columns=columns, need_col=need_col, data_li=data_li)
        return df


    async def fetch_funding_rate(
            self, 
            url, symbol, limit, columns,
            start_ts13, 
            end_ts13 = None, 
            need_col = ["datetime", "funding_rate"],
            **kwargs
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
        params = {k: v for k, v in params.items() if v is not None}     # none in params will make sesstion.get() error
        async with aiohttp.ClientSession() as session:
            while True:
                message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
                params['startTime'] = start_ts13                        # update start_time
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                data_li.extend(data) 
                if len(data) < limit:
                    break
                last = -1    
                start_ts13 = int(data[last]['fundingTime']) + 1
        df = self.form_dataframe(columns=columns, need_col=need_col, data_li=data_li)
        return df


    async def fetch_long_short_ratio(
            self, 
            url, symbol, freq, limit, columns,
            start_ts13,
            end_ts13 = None, 
            need_col = ["datetime", "long_short_ratio", "long_account", "short_account"],
            **kwargs
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
        params = {k: v for k, v in params.items() if v is not None}     # none in params will make sesstion.get() error
        async with aiohttp.ClientSession() as session:
            while True:
                message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
                params['startTime'] = start_ts13                        # update start_time
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                data_li.extend(data) 
                if len(data) < limit:
                    break
                last = -1    
                start_ts13 = int(data[last]['timestamp']) + 1
        df = self.form_dataframe(columns=columns, need_col=need_col, data_li=data_li)
        return df

