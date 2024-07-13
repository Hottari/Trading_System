import pandas as pd
from datetime import datetime
import json

import os, sys
sys.path.extend(['../', '../../'])
from prepare_data import ExchangeData
from data_processor import DataProcessor

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')

# read data
import configparser
config = configparser.ConfigParser()
for file_name in os.listdir(CONFIG_ROOT):
    file_path = os.path.join(CONFIG_ROOT, file_name)
    if os.path.isfile(file_path):
        if file_name.endswith('.ini'):
            config.read(file_path)

import tejapi
tejapi.ApiConfig.api_key = config.get('tej_api', 'api_key')


class TEJHandler(ExchangeData, DataProcessor):  
    """
        說明頁面: https://api.tej.com.tw/document_python.html
        第一季季報: 5月15日前
        第二季季報: 8月14日前
        第三季季報: 11月14日前
        年度財務報告: 次年3月31日前
    """  
    def __init__(
            self, 
            exchange='tej', symbol_type=None
        ):
        if symbol_type is not None:
            super().__init__(exchange, symbol_type)
            self.datatable_code = self.get_datatable_code()
            self.columns_rename_dict = self.get_columns_fetch()
            self.columns_datetime = self.get_columns_datetime_type()
            self.columns_string = self.get_columns_string_type()
            self.columns_sort = self.get_columns_sort()
            self.columns_duplicate = self.get_columns_duplicate()
            self.params_name = self.get_params_name()

    def get_api_using_info(self,):
        required_keys = ['reqDayLimit', 'rowsDayLimit', 'rowsMonthLimit', 'todayReqCount', 'todayRows', 'monthRows']
        df_result = pd.DataFrame([tejapi.ApiConfig.info()], index=['value'])[required_keys]
        df_result.insert(0, 'remain_req_day', df_result['reqDayLimit'] - df_result['todayReqCount'])
        df_result.insert(1, 'remain_rows_day', df_result['rowsDayLimit'] - df_result['todayRows'])
        df_result.insert(2, 'remain_rows_month', df_result['rowsMonthLimit'] - df_result['monthRows'])
        return df_result.T

    def fetch_data(self, start_date:pd.Timestamp=None, end_date:pd.Timestamp=None, symbols:list=None):
        """
            set params
            fetch data
        """
        params = {
            'datatable_code': self.datatable_code,
            'coid': symbols,
            self.params_name['datetime']: {
                'gte': start_date,#datetime.today()-pd.DateOffset(days=15) if start_date is None else start_date,
                'lte': end_date,
            },
            'opts': {'columns': list(self.columns_rename_dict.keys())},
            'paginate': True,
            # 'chinese_column_name': True       # 不要設中文比較好更名 -> 作名稱一致化
        }
        data = (
            tejapi.get(**params)
            .rename(columns=self.columns_rename_dict)
            .pipe(self.clean_data, datetime_cols=self.columns_datetime, str_cols=self.columns_string)
            .sort_values(by=self.columns_sort)
        )
        return data
    
      