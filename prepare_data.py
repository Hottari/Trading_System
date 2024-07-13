import os, sys
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(1, PROJECT_ROOT) 

class ExchangeData:
    """
    binance: see https://binance-docs.github.io/apidocs/
    okx: see https://www.okx.com/docs-v5/en/
    bybit: see https://bybit-exchange.github.io/docs/v5/intro
    tej: see https://api.tej.com.tw/index.html

    """

    def __init__(self, exchange, symbol_type, long_short_ratio_type=None):
        self.exchange = exchange
        self.symbol_type = symbol_type
        self.exchange_config = self.get_exchange_config()
        self.long_short_ratio_type = long_short_ratio_type
      
    def get_exchange_config(self):
        file_name = f"{self.exchange}.json"
        with open(os.path.join(PROJECT_ROOT, 'config', file_name)) as f:
            exchange_config = json.loads(f.read())
        return exchange_config

    # ==================== end point ==================== #
    def get_end_point(self):
        return self.exchange_config[self.symbol_type]["end_point"]
    
    def get_suffix_exchange_info(self):
        return self.exchange_config[self.symbol_type]["suffix"]["exchange_info"]
    
    def get_suffix_kline(self):
        return self.exchange_config[self.symbol_type]["suffix"]["kline"]
    
    def get_suffix_funding_rate(self):
        return self.exchange_config[self.symbol_type]["suffix"]["funding_rate"]

    def get_suffix_long_short_ratio(self):
        return self.exchange_config[self.symbol_type]["suffix"][self.long_short_ratio_type]

    # ==================== limit ==================== #
    def get_limit_kline(self):
        return self.exchange_config[self.symbol_type]["limit"]["kline"]

    def get_limit_funding_rate(self):
        return self.exchange_config[self.symbol_type]["limit"]["funding_rate"]    

    def get_limit_long_short_ratio(self):
        return self.exchange_config[self.symbol_type]["limit"][self.long_short_ratio_type] 

    # ==================== response columns ==================== #
    def get_columns_kline(self):
        return self.exchange_config[self.symbol_type]["columns"]["kline"]

    def get_columns_funding_rate(self):
        return self.exchange_config[self.symbol_type]["columns"]["funding_rate"]

    def get_columns_long_short_ratio(self):
        return self.exchange_config[self.symbol_type]["columns"][self.long_short_ratio_type]
    
    def get_columns_fetch(self):
        return self.exchange_config[self.symbol_type]["columns"]["fetch"]
    
    def get_columns_datetime_type(self):
        return self.exchange_config[self.symbol_type]["columns"]["datetime_type"]
    
    def get_columns_string_type(self):
        return self.exchange_config[self.symbol_type]["columns"]["string_type"]
    
    def get_columns_sort(self):
        return self.exchange_config[self.symbol_type]["columns"]["sort"]

    def get_columns_uncategorized(self):
        return self.exchange_config[self.symbol_type]["columns"]

    # ==================== params name ==================== #
    def get_params_name(self):
        return self.exchange_config[self.symbol_type]["params_name"]
    
    # ==================== others ==================== #
    def get_datatable_code(self):
        return self.exchange_config[self.symbol_type]["datatable_code"]
    