import pandas as pd
import numpy as np
import requests, time, asyncio # aiohttp
import importlib

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.extend([PROJECT_ROOT, '../..']) 
from prepare_data import ExchangeData
from message_manager import MessageManager
message = MessageManager()


class DBLoader():
    def __init__(self):
        pass

    def get_db_df(
            self, 
            collection_name, db, db_name=None, 
            need_col=None, query_dict=None, sort_name=None,
            limit=0,
        ):
        collection = db[db_name][collection_name] if db_name else db[collection_name]
        # filter
        projection_dict = {'_id': 0}
        projection_dict.update({col: 1 for col in need_col}) if need_col else None
        query = collection.find(query_dict, projection_dict)
        # sort
        query = query.sort(sort_name, -1) if sort_name else query
        data = query.limit(limit)
        df = pd.DataFrame(list(data))
        return df


class DataLoader(ExchangeData):
    def __init__(
            self, 
            exchange, symbol_type=None, start=None, end=None, timezone='Asia/Taipei',
            long_short_ratio_type='global',
        ):
        super().__init__(exchange, symbol_type, long_short_ratio_type)
        self.timezone = timezone
        self.start_ts13 = self.get_ts13(start) if start is not None else 0
        self.end_ts13 = self.get_ts13(end) if end is not None else int(time.time()*1000)
    
    # ==================== Basic ==================== # 
    def get_ts13(self, datetime_str:str,):
        ts13 = int(pd.to_datetime(datetime_str).tz_localize(self.timezone).timestamp()*1000)
        return ts13

    def update_dic(self, dic_ori, update_values):
        """  get update but not change ori_dict """
        dic = dic_ori.copy()
        dic.update(update_values) if update_values else dic
        return dic
    
    # ==================== Fetch ==================== #
    def get_all_symbol_list(self,):
        url_exchange_info = self.get_end_point() + self.get_suffix_exchange_info()
        exchange_info = requests.get(url_exchange_info).json()
        symbol_li = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']
        return symbol_li
    
    def get_fetch_args(self, item:str):
        end_point = self.get_end_point()
        if item == 'ohlcv': 
            url = end_point + self.get_suffix_kline()
            limit = self.get_limit_kline()
            columns = self.get_columns_kline()

        elif item == 'funding_rate': 
            url = end_point + self.get_suffix_funding_rate()
            limit = self.get_limit_funding_rate()
            columns = self.get_columns_funding_rate()

        elif item == 'long_short_ratio': 
            url = end_point + self.get_suffix_long_short_ratio()
            limit = self.get_limit_long_short_ratio()
            columns = self.get_columns_long_short_ratio()
        else:
            raise ValueError(f"Unsupported item: {item}")

        fetch_args = {
            'url': url,
            'limit': limit,
            'columns': columns,
        }
        return fetch_args

    def get_fetch_function(self, item:str):
        exchange_handlers = {
            'binance': 'data_loader.api_connecter.binance_handler.BinanceHandler',
            'okx': 'data_loader.api_connecter.okx_handler.OKXHandler',
            # Add more exchanges here...
        }
        if self.exchange in exchange_handlers:
            module_path, class_name = exchange_handlers[self.exchange].rsplit('.', 1)
            module = importlib.import_module(module_path)
            handler_class = getattr(module, class_name)

            # get instance -> fetch_func is an instance method rather than class method
            handler_instance = handler_class()
            if item == 'ohlcv': 
                return handler_instance.fetch_ohlcv
            elif item == 'funding_rate': 
                return handler_instance.fetch_funding_rate
            elif item == 'long_short_ratio': 
                return handler_instance.fetch_long_short_ratio
            else:
                raise ValueError(f"Unsupported item: {item}")
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")


    async def do_fetch_update(
    # def do_fetch_update(
            self, 
            save_dir, item, freq:str=None,
            symbol_li:dict=None,
        ):
        fetch_func = self.get_fetch_function(item)
        os.makedirs(save_dir, exist_ok=True)

        start_ts13 = self.start_ts13
        end_ts13 = self.end_ts13
        if item in ['long_short_ratio']:
            # long_short_ratio limit: 30 days
            start_ts13 = max(end_ts13-30*24*60*60*1000, start_ts13) if end_ts13 else max(int(time.time()*1000)-30*24*60*60*1000, start_ts13)
        fetch_args = self.get_fetch_args(item)
        fetch_args.update(
            start_ts13 = start_ts13,
            end_ts13 = end_ts13,
            freq = freq,
        )

        if not symbol_li:   
            symbol_li = self.get_all_symbol_list()
        
        symbol_amount = len(symbol_li)
        for symbol in symbol_li:
            await asyncio.sleep(3)                        # avoid hitting request rate limit
            message.updating(item, symbol_li.index(symbol)+1, symbol_amount)
            file_path = os.path.join(save_dir, f"{symbol}_{item}.pkl")
            df_ori = pd.read_pickle(file_path) if os.path.isfile(file_path) else pd.DataFrame()
            fetch_args['symbol'] = symbol

            try:
                # no file or empty -> create new one
                if not os.path.isfile(file_path) or df_ori.empty:
                    message.necessary_creating(item, symbol)
                    df_updated = await fetch_func(**fetch_args)
                    # df_updated = fetch_func(**fetch_args)

                    if df_updated.empty:
                        message.empty_warning(item, symbol) 
                    else:
                        df_updated = df_updated.map(lambda x: pd.to_numeric(x, errors='coerce')).astype('float64')
                        df_updated.to_pickle(file_path)
                        message.donload_completed(item, symbol, file_path)

                # file exists and not empty -> update
                elif not df_ori.empty:          
                    start_ts13_ori = int(df_ori.index[0].timestamp()*1000)
                    end_ts13_ori = int(df_ori.index[-1].timestamp()*1000)
                    update_li = [df_ori]

                    if (start_ts13 > start_ts13_ori) & (end_ts13 < end_ts13_ori):
                        message.unnecessary_update(item, symbol)
                        continue
                    # update head
                    if start_ts13 < start_ts13_ori:
                        df_new_head = await fetch_func(**self.update_dic(fetch_args, {'end_ts13': start_ts13_ori-1}))
                        # df_new_head = fetch_func(**self.update_dic(fetch_args, {'end_ts13': start_ts13_ori-1}))
                        update_li.insert(0, df_new_head.map(lambda x: pd.to_numeric(x, errors='coerce')).astype('float64')) 
                    # update tail
                    if end_ts13 > end_ts13_ori:
                        df_new_tail = await fetch_func(**self.update_dic(fetch_args, {'start_ts13': end_ts13_ori+1}))
                        # df_new_tail = fetch_func(**self.update_dic(fetch_args, {'start_ts13': end_ts13_ori+1}))
                        update_li.append(df_new_tail.map(lambda x: pd.to_numeric(x, errors='coerce')).astype('float64'))
                    df_updated = pd.concat(update_li, axis='rows')
                    df_updated.to_pickle(file_path)
                    message.update_completed(item, symbol)

            except Exception as e:
                message.symbol_error(item, symbol)
                continue


