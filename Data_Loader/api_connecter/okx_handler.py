import pandas as pd
import numpy as np
import os, sys
import requests, time, aiohttp

PROJECT_ROOT = '..'
sys.path.extend([PROJECT_ROOT, '../..']) 
from message_manager import MessageManager
message = MessageManager()

class OKXHandler(ExchangeData):

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
        - spot, usd: see https://www.okx.com/docs-v5/en/?shell#order-book-trading-market-data-get-candlesticks-history
        - coin: see 
        """
        item = 'ohlcv'
        data_li = []
        params = {
            'instId': symbol, 
            'bar': freq, 
            'limit':limit,
            'after': end_ts13,
        }
        params = {k: v for k, v in params.items() if v is not None}     # none in params will make sesstion.get() error
        async with aiohttp.ClientSession() as session:
            while True:
                message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
                params['before'] = start_ts13                           # update start_time
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                data_li.extend(data) 
                if len(data) < limit:
                    break
                last = 0    
                start_ts13 = int(data['data'][last][0]) + 1             # columns 0 is 'ts'
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
        - usd: see https://www.okx.com/docs-v5/en/?python#public-data-rest-api-get-funding-rate-history
        """
        item = 'funding_rate'
        data_li = []
        params = {
            'instId': symbol, 
            'limit':limit,
            'after': end_ts13,
        }
        params = {k: v for k, v in params.items() if v is not None}     # none in params will make sesstion.get() error
        async with aiohttp.ClientSession() as session:
            while True:
                message.fetching(item, symbol, start_time=pd.to_datetime(start_ts13, unit='ms').tz_localize('UTC'))
                params['before'] = start_ts13                           # update start_time
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                data_li.extend(data) 
                if len(data) < limit:
                    break
                last = 0    
                start_ts13 = int(data['data'][last]['fundingTime']) + 1
        df = self.form_dataframe(columns=columns, need_col=need_col, data_li=data_li)
        return df





