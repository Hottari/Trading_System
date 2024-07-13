import numpy as np
import pandas as pd

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import dask.dataframe as dd

class DataProcessor():
    def __init__(self):
        pass

    # ========================== data type ========================== #
    def clean_data(self, data:pd.DataFrame, datetime_cols=[], str_cols=[]):
        for col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            if col in datetime_cols:
                data[col] = pd.to_datetime(data[col]).dt.tz_localize(None)
            elif col not in str_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce') 
        return data
    
    # ========================== file type ========================== #
    def get_parquet_columns(self, data_path)->list:
        """
            Get parquet file columns
            
            Args:
            - data_path (str): parquet file path

            Returns:
            - list: parquet file columns
        """
        return pq.ParquetFile(source=data_path).schema.names

    def get_parquet_data_with_dask(
            self,
            data_path:str, 
            datetime_col:str= 'datetime', 
            start_date = None, end_date = None,
            need_cols = None, 
        )-> pd.DataFrame:
        # set filter conditions
        # datetime filter only temporarily
        filters = []
        if start_date is not None:
            filters.append((datetime_col, '>=', pd.to_datetime(start_date)))
        if end_date is not None:
            filters.append((datetime_col, '<=', pd.to_datetime(end_date)))
        specific_partition_df = dd.read_parquet(data_path, filters=filters, columns=need_cols)
        if need_cols is not None:
            if datetime_col in need_cols:
                filtered_df = specific_partition_df.groupby(datetime_col).sum().compute()
            else: 
                filtered_df = specific_partition_df.compute()
        else:
            filtered_df = specific_partition_df.compute()
        return filtered_df

    def get_parquet_data_with_pyarrow(
            self, 
            data_path:str, 
            datetime_col:str= 'datetime', 
            start_date = None, end_date = None,
            need_cols = None,
        )-> pd.DataFrame:
        """
            Get speific datetime period and columns data from parquet file
            
            Args:
            - data_path (str): parquet file path
            - datetime_col (str): datetime column name
            - start_date (pd.Timestamp): start date, default None
            - end_date (pd.Timestamp): end date, default None

            Returns:
            - pd.DataFrame: dataframe with specific datetime period and columns data
        """
        # set filter conditions
        filters = []
        if start_date is not None:
            filters.append(ds.field(datetime_col) >= pa.scalar(start_date))
        if end_date is not None:
            filters.append(ds.field(datetime_col) <= pa.scalar(end_date))

        combined_filter = filters[0] if filters else None
        for f in filters[1:]:
            combined_filter = combined_filter & f

        # get data
        dataset = ds.dataset(data_path, format="parquet")
        filtered_table = dataset.to_table(
            columns = need_cols,
            filter = combined_filter
        )
        filtered_df = filtered_table.to_pandas()
        return filtered_df

    def update_parquet_data_with_dask(
            self, 
            data_path:str,
            new_data:pd.DataFrame,
            keep:str,
            sort_columns:list = ['datetime', 'symbol'],
            duplicate_columns:list = ['datetime', 'symbol'],
            
        ):
        """
            Replace the overlapping data with new data and add missing columns.

            Args:
            - data_path (str): parquet file path
            - new_data (pd.DataFrame): new data
            - datetime_column (str): datetime column name
        """
        # get new and non-overlapping table
        existing_ddf = dd.read_parquet(data_path)
        # faster than "npartitions=1"
        new_ddf = dd.from_pandas(new_data, npartitions=existing_ddf.npartitions)
        combined_ddf = dd.concat([existing_ddf, new_ddf])
        if set(sort_columns) <= set(combined_ddf.columns):
            updated_ddf = (
                combined_ddf
                .sort_values(by=sort_columns)
                .drop_duplicates(subset=duplicate_columns, keep=keep)
            )
        else:
            raise ValueError(f"sort_columns should be subset of {combined_ddf.columns}")
        # dask 無法複寫 parquet, or 必須先刪除再複寫 -> 風險極大
        updated_table = pa.Table.from_pandas(updated_ddf.compute())
        pq.write_table(updated_table, data_path, compression='snappy')

    def add_missing_columns_to_paTable_with_none(self, table:pa.Table, missing_columns, reference_schema):
        for col in missing_columns:
            data_type = reference_schema.field(col).type
            none_column = pa.array([None] * table.num_rows, type=data_type)
            table = table.append_column(col, none_column)
        return table

    def update_parquet_data_with_pyarrow(
            self, 
            data_path:str,
            new_data:pd.DataFrame, 
            datetime_column:str = 'datetime',
        ):
        """
            Replace the overlapping data with new data and add missing columns.

            Args:
            - data_path (str): parquet file path
            - new_data (pd.DataFrame): new data
            - datetime_column (str): datetime column name
        """
        # get new and non-overlapping table
        new_table = pa.Table.from_pandas(new_data, preserve_index=False)
        existing_table = ds.dataset(data_path, format="parquet")
        datetime_sorted = new_data[datetime_column].sort_values()
        start_date = datetime_sorted.iloc[0]
        end_date = datetime_sorted.iloc[-1]
        non_overlapping_table = existing_table.to_table(
            columns = None,
            filter = ( ds.field(datetime_column) < pa.scalar(start_date)) | (ds.field(datetime_column) > pa.scalar(end_date))
        )
        new_table_schema = new_table.schema
        non_overlapping_table_schema = non_overlapping_table.schema

        # add missing columns
        missing_in_new_table = set(non_overlapping_table_schema.names) - set(new_table_schema.names)
        missing_in_non_overlapping_table = set(new_table_schema.names) - set(non_overlapping_table_schema.names)
        new_table = self.add_missing_columns_to_paTable_with_none(new_table, missing_in_new_table, non_overlapping_table_schema)
        non_overlapping_table = self.add_missing_columns_to_paTable_with_none(non_overlapping_table, missing_in_non_overlapping_table, new_table_schema)
        
        # schema order and type
        # new_table = new_table.select(non_overlapping_table_schema.names)
        new_schema = pa.schema([(field.name, field.type) for field in non_overlapping_table.schema])
        new_schema_field_names = [field.name for field in new_schema]
        ## make same order
        new_table = new_table.select(new_schema_field_names)
        ## make same type
        new_table = new_table.cast(new_schema)

        updated_table = pa.concat_tables([non_overlapping_table, new_table])
        pq.write_table(updated_table, data_path, compression='snappy')


    # ========================== index ========================== #
    def add_time(self, df, datetime_name='datetime'):
        df['year'] = df[datetime_name].dt.year
        df['month'] = df[datetime_name].dt.month
        df['hour'] = df[datetime_name].dt.hour
        df['weekday'] = df[datetime_name].dt.weekday + 1
        df['time'] = df.hour + df[datetime_name].dt.minute/100
        df['date'] = pd.to_datetime(df[datetime_name].dt.date)
        return df
    
    # ========================== ohlcvr ========================== #
    def get_ret_oc(self, df):
        return df['close']/df['open']-1

    def get_ret_oo(self, df):
        ret_oc = self.get_ret_oc(df)
        ret_oo = (1+df['ret'])/((1+ret_oc)/(1+ret_oc.shift(1))) - 1
        return ret_oo
    
    def get_ret_co(self, df):
        ret_oc = self.get_ret_oc(df)
        ret_co = (1+df['ret']) / (1+ret_oc) - 1
        return ret_co

    def resample_ohlcv(
            self, 
            df: pd.DataFrame, 
            freq: str, 
            ohlcv_li: list = ['open', 'high', 'low', 'close', 'volume'], 
            exchange: str = None):
        """
        Get resampleed ohlcv.

        Args:
        - df (pd.DataFrame): dataframe with ohlcv and datetime type index.
        - freq: (str): resample frequence.
        - ohlcv_li (list): ohlcv column name list.
        - exchange (str): data source ( from which exchange )  

        Returns:
            pd.DataFrame: resampled dataframe with ohlcv only.

        Example:
            df_resampled = resample_ohlcv(df, freq, ohlcv_li=['open', 'high', 'low', 'close', 'volume'], exchange='binance)
        """
        if not exchange:
            closed, label = 'right', 'right'
        elif exchange=='binance':  
            closed, label = 'left', 'left'
        imply_li = ['first', 'max', 'min', 'last', 'sum']
        agg_funcs = dict(zip(ohlcv_li, imply_li))
        df_resampled = df.resample(freq, closed=closed, label=label).agg(agg_funcs)
        return df_resampled

    def add_adj_ohlc(self, df_rohlc:pd.DataFrame, is_c_only:bool=False)->pd.DataFrame:
        """
        Get adjusted price.

        Args:
        - df (dataframe): dataframe with ret, un-rejusted OHLC 

        Returns:
            dataFrame: original df add adjusted price columns

        Example:
            df_adj = df.groupby('symbol').apply(add_adj_ohlc, is_c_only=True).droplevel(0)
        
        Mind:
        - columns 'ret', 'close' are necessary in df
        """
        df = df_rohlc.copy()
        ret_comprod_reverse = (1 + df['ret']).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1) # adj from t-1
        close_adj = np.round(df['close'].iloc[-1] / ret_comprod_reverse, 3)                       # Close[-1] is the base price
        
        if not is_c_only:
            df['adj_rate'] = close_adj/df['close']
            df['open_adj'] = np.round(df['open'] * df['adj_rate'], 3)
            df['high_adj'] = np.round(df['high'] * df['adj_rate'], 3)
            df['low_adj'] = np.round(df['low'] * df['adj_rate'], 3)
        
        df['close_adj'] = close_adj
        return df

    # ========================== time series ========================== #
    def get_rolling_ema(self, data, window, alpha):
        ema_data = np.full(window-1, np.nan)
        for i in range(window-1, len(data)):
            ema_data = np.append(
                ema_data, 
                data[i-window+1:i+1].ewm(alpha=alpha, adjust=False).mean().values[-1]
            )
        return ema_data

    def add_yoy(self, df_data:pd.DataFrame, data_name_li:list):
        """
        Add yoy to DataFrame.

        Args:
        - df_data (pd.DataFrame): DataFrame to add yoy columns.
        - data_name_li (list): List of column names to calculate yoy.

        Example:
            get_factor = GetFactor()
            df_result = get_factor.add_yoy(df_data, data_name_li)
        
        Mind: 
        - There should be a year with quarter, or month, or any other time period in the index.
            - ex. ['year', 'month'] or ['year', 'quarter']
        - keep NaN ( without dropna() ).  
        
        """
        df = df_data.copy()
        index_col = df.index.names
        df_pre = df.reset_index()
        df_pre['year'] += 1
        for data_name in data_name_li:
            df[f"{data_name}_last_year"] = df_pre.set_index(index_col)[data_name]
            df[f"{data_name}_yoy"] = df[data_name]/df[f"{data_name}_last_year"]-1
        return df

    def add_rolling_stand(
            self, 
            df_data:pd.DataFrame, 
            data_name_li:list, 
            window:int = 4,
            past_period:int = 1,
        ):
        """
        Add rolling_stand(_stand) to DataFrame.

        Args:
        - df_data (pd.DataFrame): DataFrame to add yoy columns.
        - data_name_li (list): List of column names to calculate yoy.
        - window (int): rolling window of stand
        - past_period (int): past period of stand

        Example:
        - get_factor = GetFactor()
        - df_result = get_factor.add_rolling_stand(df_data, data_name_li)        
        
        Mind:
        - There should be 'symbol' in index. 
             - ex. ['year', 'month', 'symbol']
        """
        df = df_data.unstack().sort_index().copy() # set symbol index to the column
        for data_name in data_name_li:
            df_mean = df[data_name].rolling(window).mean().shift(past_period)
            df_std = df[data_name].rolling(window).std().shift(past_period)
            df_stand = (df[data_name] - df_mean)/df_std
            df_data[f"{data_name}_stand"] = df_stand.stack()
        return df.stack()
    

# ==================================== Tail ==================================== #
