import numpy as np
import pandas as pd

def get_adj_price(df:pd.DataFrame)->pd.DataFrame:
    """
    Get adjusted price.

    Args:
        df (pd.DataFrame): dataframe with Ret, un-rejusted OHLC 

    Returns:
        pd.DataFrame: original df add adjusted price columns

    Example:
        df_adj = df_ori.groupby('Code', group_keys=False).apply(get_adj_price)
    """

    ret_comprod_reverse = (1 + df['Ret']).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1) # adj from t-1
    close_adj = np.round(df['Close'].iloc[-1] / ret_comprod_reverse, 3)                       # Close[-1] is the base price
    df['Adj_rate'] = close_adj/df['Close']
    df['Open_adj'] = np.round(df['Open'] * df['Adj_rate'], 3)
    df['High_adj'] = np.round(df['High'] * df['Adj_rate'], 3)
    df['Low_adj'] = np.round(df['Low'] * df['Adj_rate'], 3)
    df['Close_adj'] = close_adj
    return df



def resample_ohlcv(df, freq, ohlcv_li=['open', 'high', 'low', 'close', 'volume'], exchange=None):
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
    