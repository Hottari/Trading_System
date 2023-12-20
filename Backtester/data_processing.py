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
    