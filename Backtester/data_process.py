import numpy as np
import pandas as pd


def add_time(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday + 1
    df['month'] = df.index.month
    df['time'] = df.index.hour + df.index.minute/100
    df['date'] = pd.to_datetime(df.index.date)
    return df

class OHLCVProcess():
    def __init__(self):
        pass

    def resample_ohlcv(
            self, 
            df: pd.DataFrame, 
            freq: str, 
            ohlcv_li: list = ['open', 'high', 'low', 'close', 'volume'], 
            exchange: str = None):
        """
        Get resampleed ohlcv.

        Args:
            df (pd.DataFrame): dataframe with ohlcv and datetime type index.
            freq: (str): resample frequence.
            ohlcv_li (list): ohlcv column name list.
            exchange (str): data source ( from which exchange )  

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


    def add_ret(self, df):
        """
        Get different ret.

        Args:
            df (pd.DataFrame): dataframe with columns 'open', 'close' 

        Returns:
            pd.DataFrame: original df add different ret columns

        Example:
            df_ret_added = add_ret(df)
        
        Mind:
        - columns 'open', 'close' are necessary in df
        """

        df['ret_cc'] = df['close'].pct_change()
        df['ret_oc'] = df['close']/df['open'] -1
        df['ret_oo'] = df['open'].pct_change()
        df['ret_co'] = df['open']/df['close'].shift(1) -1
        return df


    def add_adj_ohlc(self, df_rohlc:pd.DataFrame)->pd.DataFrame:
        """
        Get adjusted price.

        Args:
            df (pd.DataFrame): dataframe with Ret, un-rejusted OHLC 

        Returns:
            pd.DataFrame: original df add adjusted price columns

        Example:
            df_adj = df_ori.groupby('Code', group_keys=False).apply(add_adj_ohlc)
        
        Mind:
        - columns 'ret', 'close' are necessary in df
        """
        df = df_rohlc.copy()
        ret_comprod_reverse = (1 + df['ret']).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1) # adj from t-1
        close_adj = np.round(df['close'].iloc[-1] / ret_comprod_reverse, 3)                       # Close[-1] is the base price
        df['adj_rate'] = close_adj/df['close']
        df['open_adj'] = np.round(df['open'] * df['adj_rate'], 3)
        df['high_adj'] = np.round(df['high'] * df['adj_rate'], 3)
        df['low_adj'] = np.round(df['low'] * df['adj_rate'], 3)
        df['close_adj'] = close_adj
        return df



class GetFactor():
    def __init__(self, ):
        pass

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
        - There should be a year and quarter, or month, or another time period in the index.
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
        Add yoy to DataFrame.

        Args:
        - df_data (pd.DataFrame): DataFrame to add yoy columns.
        - data_name_li (list): List of column names to calculate yoy.

        Example:
            get_factor = GetFactor()
            df_result = get_factor.add_yoy(df_data, data_name_li)
        
        Mind: There should be a year and quarter, or month, or another time period in the index.
        
        """
        df = df_data.copy()
        mean_name = f"mean_current{window}_past{past_period}"
        std_name = f"std_current{window}_past{past_period}"

        for data_name in data_name_li:
            data = df[data_name].values
            mean = ( df[data_name].rolling(window).mean().shift(past_period) ).values
            std = ( df[data_name].rolling(window).std().shift(past_period) ).values
            stand = ( data-mean )/std
            
            df[f"{data_name}_{mean_name}"] = mean
            df[f"{data_name}_{std_name}"] = std
            df[f"{data_name}_stand"] = stand

        return df