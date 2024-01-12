import numpy as np
import pandas as pd
from numba import jit

class BackTest():
    def __init__(self):
        pass

    # return
    #@jit(nopython=True)
    def get_ret(self, ret_arr:np.ndarray, friction_cost_arr:np.ndarray, signal_arr:np.ndarray, LS_adjust:int=1)-> dict: 
        """
        讀取進出場訊號來計算報酬 (不考慮加碼)

        Args:
            ret_arr (np.ndarray): 買進持有報酬
            friction_cost_arr (np.ndarray): 當期摩擦成本 
            signal_arr (np.ndarray): 進出場訊號
            LS_adjust (int): 多空單報酬方向調整, 多=1 空=-1

        Returns:
            dict: 部位, 報酬

        Example:
            get_ret(ret_arr, friction_cost_arr, signal_short_arr, LS_adjust=-1)
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
                    strategy_ret_arr[t] = ret_arr[t]*LS_adjust - friction_cost_arr[t]
                    
            elif pos:                                   # 有部位時：
                if signal_arr[t-1] == -1:               # 前一期有出場訊號 -> 更新部位; 原持有報酬視為出場報酬 -> 原報酬-摩擦成本
                    pos = False
                    strategy_ret_arr[t-1] -= friction_cost_arr[t]

                else: strategy_ret_arr[t] = ret_arr[t]*LS_adjust  # 否則僅更新持有報酬
        
        result = {
            'pos_arr':pos_arr,
            'strategy_ret_arr':strategy_ret_arr,
            }
        return result


    def ret_from_signal(self, df:pd.DataFrame, is_long:bool=True, is_short:bool=True)-> dict: 
        """
        分別讀取多空進出場訊號來計算多空報酬

        Args:
            df (pd.DataFrame): 包含報酬, 多空訊號, 摩擦成本
            is_long (bool): 是否做多
            is_short (bool): 是否做空

        Returns:
            dict: 多單部位, 多單報酬, 空單部位, 空單報酬

        Example:
            ret_from_signal(df, is_short=False)
        """

        ret_arr = df['ret'].shift(-1).fillna(0).to_numpy()   # 持有到下一期才產生報酬
        friction_cost_arr = df['friction_cost'].to_numpy()
        
        T = len(ret_arr)
        pos_long_arr = pos_short_arr = strategy_ret_long_arr = strategy_ret_short_arr = np.zeros(T) # initial pos_arr, ret_arr in case no long or no short

        if is_long: 
            signal_long_arr = df['signal_long'].to_numpy()
            pos_long_arr, strategy_ret_long_arr = self.get_ret(
                ret_arr = ret_arr, 
                friction_cost_arr = friction_cost_arr, 
                signal_arr = signal_long_arr, 
                LS_adjust = 1,
            ).values()
        if is_short:
            signal_short_arr = df['signal_short'].to_numpy()
            pos_short_arr, strategy_ret_short_arr = self.get_ret(
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



    # performance
    def perf_TMBA(self, ret_ts:pd.Series, annual_factor=252):
        ret_cum = ret_ts.fillna(0).cumsum()                  # in case NaN

        total_return = ret_cum[-1]
        annual_return = ret_ts.mean()*annual_factor
        profit_ts = ret_ts[ret_ts>0]
        loss_ts = ret_ts[ret_ts<0]
        profit = profit_ts.cumsum()[-1]
        loss = loss_ts.cumsum()[-1]
        win_times = profit_ts.shape[0]
        lose_times = loss_ts.shape[0]

        sharpe = ret_ts.mean() / ret_ts.std() * np.sqrt(annual_factor)
        mdd = (ret_cum - np.maximum.accumulate(ret_cum) ).min()
        ret_to_risk = total_return/np.abs(mdd)
        profit_to_loss = profit/(-loss)

        try: win_rate = win_times / (win_times + lose_times)  # in case no enter
        except: win_rate = 0
        
        perf_dict = {
            'Total_Return(%)': total_return*100,
            'Annual_Return(%)': annual_return*100,
            'Annnal_Sharpe': sharpe,
            'MDD(%)': mdd*100,
            'Ret_to_Risk': ret_to_risk, 
            'profit_to_loss': profit_to_loss,
            #'aveg_holding_period': aveg_holding_period,
            'Win_Rate': win_rate,
            #'Total_Trades': pf.trades.count(),
        }

        perf_dict_rounded = dict(zip(
            perf_dict.keys(), 
            np.round( list(perf_dict.values()), 2)
            ))

        return perf_dict_rounded
