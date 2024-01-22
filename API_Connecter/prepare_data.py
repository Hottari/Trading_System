import os, sys
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(1, PROJECT_ROOT) 

class ExchangeData:

    def __init__(self, exchange, symbol_type):
        self.exchange = exchange
        self.symbol_type = symbol_type
        self.exchange_config = self.get_exchange_config()
      
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
    

    # ==================== limit ==================== #

    def get_limit_kline(self):
        return self.exchange_config[self.symbol_type]["limit"]["kline"]
    

    # ==================== columns ==================== #

    def get_columns_kline(self):
        return self.exchange_config[self.symbol_type]["columns"]["kline"]