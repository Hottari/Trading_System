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

    def __init__(self, timezone='Asia/Taipei'):
        self.spot_client = ccxt.binance()
        self.timezone = timezone
        #pass

    def download_spot_ohlcv(self, start, freq, spot_li=None, limit=1000):
        """
        No check if_spot in spot_li
        """

        if not spot_li:
            api_url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(api_url)
            exchange_info = response.json()
            spot_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']
        
        # freq = '1m'
        # start = '2023-12-01 00:00:00+08:00'
        # spot_li = ['BTCUSDT']
        # df_all = self.fetch_all_ohlcv_data(self.spot_client, spot_li, freq, start)
            
        df_li = [self.fetch_spot_ohlcv(self.spot_client, symbol, start, freq).set_index(['date', 'code']) for symbol in spot_li]

        #TODO: 每個幣載完就存, 存到哪裡, name=BTCUSDT_ohlcv.pkl


        df_all = pd.concat(df_li, axis='rows').reset_index()
        df_all['date'] = pd.to_datetime(df_all['date'], unit='ms').dt.tz_localize('UTC').tz_convert(self.timezone)
        df_all = df_all.set_index(['date', 'code']).sort_index()
        
        path = os.path.dirname(__file__)

        save_dir = f'{os.path.dirname(__file__)}/../backtest_result/{self._strategy_class}/{self._strategy_config}'
        os.makedirs(save_dir, exist_ok=True)
        position_df.to_csv(f'{save_dir}/{self._strategy_config}_position_vbt.csv')

        df_all.to_pickle()



    def fetch_spot_ohlcv(self, symbol, start, freq, limit=1000):
        ohlcv_list = []
        start_ts13 = self.spot_client.parse8601(start)

        columns = ['date', 'open', 'high', 'low', 'close', 'volume']

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

    def fetch_all_ohlcv_data(exchange, symbol_list, freq, start):
        df_li = [self.fetch_ohlcv_data(exchange, symbol, start, freq).set_index(['date', 'code']) for symbol in symbol_list]
        df_all = pd.concat(df_li, axis='rows').reset_index()
        df_all['date'] = pd.to_datetime(df_all['date'], unit='ms').dt.tz_localize('UTC').tz_convert('Asia/Taipei')
        df_all = df_all.set_index(['date', 'code']).sort_index()
        
        return df_all








    