import numpy as np
from numba import jit

@jit(nopython=True)
def get_ret(ret_arr:np.ndarray, friction_cost_arr:np.ndarray, signal_arr:np.ndarray, LS_adjust:int=1)-> dict: 
    """
    讀取進出場訊號來計算報酬 (不考慮加碼)

    Args:
        ret_arr (np.ndarray): 買進持有報酬
        friction_cost_arr (np.ndarray): 當期摩擦成本 
        signal_arr (np.ndarray): 進出場訊號
        LS_adjust (int): 多空單報酬方向調整, 多=1 空=-1

    Returns:
        dict: 多單部位, 多單報酬, 空單部位, 空單報酬

    Example:
        get_ret(ret_arr, friction_cost_arr, signal_arr, LS_adjust=1)
    """

    T = len(ret_arr)
    pos = False
    pos_arr = np.full(T, np.nan)
    strategy_ret_arr = np.zeros(T)


    for t in range(1, T):
        if not pos:                                 # 無部位時：
            if signal_arr[t-1] == 1:                # 前一期有進場訊號 -> 更新部位; 新增持有報酬-摩擦成本
                pos = True 
                pos_arr[t] = pos
                strategy_ret_arr[t] = ret_arr[t] * LS_adjust - friction_cost_arr[t]
                
        elif pos:                                   # 有部位時：
            if signal_arr[t-1] == -1:               # 前一期有出場訊號 -> 更新部位; 原持有報酬視為出場報酬 -> -摩擦成本
                pos = False
                strategy_ret_arr[t-1] -= friction_cost_arr[t]

            else: strategy_ret_arr[t] = ret_arr[t]  # 否則更新持有報酬
    
    result = {
        'pos_arr':pos_arr,
        'strategy_ret_arr':strategy_ret_arr,
        }
    return result


@jit(nopython=True)
def ret_from_signal(df)-> dict: 

    # def ret from signal
    ret_arr = df['ret'].shift(-1).fillna(0).to_numpy()   # 持有到下一期才產生報酬
    signal_long_arr = df['signal_long'].to_numpy()
    signal_short_arr = df['signal_short'].to_numpy()
    friction_cost_arr = df['friction_cost'].to_numpy()

    pos_long_arr, strategy_ret_long_arr = get_ret(
        ret_arr = ret_arr, 
        friction_cost_arr = friction_cost_arr, 
        signal_arr = signal_long_arr, 
        LS_adjust = 1,
    ).values()

    pos_short_arr, strategy_ret_short_arr = get_ret(
        ret_arr = ret_arr, 
        friction_cost_arr = friction_cost_arr, 
        signal_arr = signal_short_arr, 
        LS_adjust = -1,
    ).values()

    result = {
        'pos_long_arr': pos_long_arr, 
        'strategy_ret_long_arr': strategy_ret_long_arr,
        'pos_short_arr': pos_short_arr, 
        'strategy_ret_short_arr': strategy_ret_short_arr,
    }
    return result