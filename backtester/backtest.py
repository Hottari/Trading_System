import numpy as np
import pandas as pd
from tabulate import tabulate
# from numba import jit

class BackTester():
    def __init__(self):
        pass

    def get_daily_ret(self, ret):
        """
        轉換日內報酬為日報酬

        Args:
            ret (pd.Series): return time series

        Return:
            pd.Series: 日報酬

        Example:
            get_daily_ret(ret)
        """
        df = ret.copy()
        return (1 + df).resample('D').prod() - 1

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


    def get_ret_from_signal(self, df:pd.DataFrame, is_long:bool=True, is_short:bool=True)-> dict: 
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
    

    def get_ret_after_fric(
            self, 
            rets, weights, 
            fr=0, fee_rate_in=0.001425*0.7, fee_rate_out=0.001425*0.3+0.003, slip_rate=0, 
            signal_delay_periods=0,
    ):
        """
        Get Return After Friction

        Args:
            rets (pd.DataFrame): return DataFrame with datetime index and symbol columns
            weights (pd.DataFrame): weighting DataFrame with datetime index and symbol columns

        Return:
            pd.DataFrame: return after friction DataFrame with datetime index and symbol columns

        Example:
            get_ret_after_fric(rets, weights)
        """
        # ret_after_fric
        weight_change = weights.diff().shift(signal_delay_periods).fillna(0)
        weights_ret_happen = weights.shift(signal_delay_periods+1).fillna(0)
        ret_weighted = weights_ret_happen * rets
        # t=0 生訊號, 立即進場生摩擦
        # t=1 生報酬

        ret_to_fric_adjust = weights_ret_happen.where(weights_ret_happen==0, weights_ret_happen/weights_ret_happen) * rets  # 有產生報酬才需要對摩擦損失做報酬調整 

        fr_loss = weights_ret_happen * fr * (1+ret_to_fric_adjust)                                                          # ret 計在下期
        fee_loss_in = ( weight_change[weight_change>0] * ( fee_rate_in * ( 1+ret_to_fric_adjust ) ) ).fillna(0)             # fee 計在轉倉當下
        fee_loss_out = -( weight_change[weight_change<0] * ( fee_rate_out * ( 1+ret_to_fric_adjust ) ) ).fillna(0)
        slip_lose = np.abs(weight_change) * ( slip_rate * ( 1+ret_to_fric_adjust ) )                                        # slippage 計在轉倉當下
        
        ret_after_fric = ret_weighted - fr_loss - fee_loss_in - fee_loss_out - slip_lose
        return ret_after_fric


    # performance
    def get_perf_table(self, ret:pd.Series, annual_factor:float=252, is_compound:bool=True, name:str='None'):
        """
        Get Performance Table

        Args:
            ret (pd.Series): return time series
            annual_factor (float): annual periods
            is_compound (bool): if compound return
            name (str): strategy name

        Return:
            pd.DataFrame: Performance Table
        """        
        ret_ts = ret.fillna(0).copy()    # incase error value compute
        if is_compound: 
            ret_cum = (1 + ret).cumprod() -1    
        else:
            ret_cum = ret_ts.cumsum()

        total_return = ret_cum.values[-1]
        cagr = ((ret_cum.values[-1] +1) ** (1/len(ret_cum))) ** annual_factor -1

        ts_dd = (ret_cum +1) / (np.maximum(0, ret_cum.cummax()) +1) - 1
        mdd = np.abs(ts_dd.min())

        ret_diff = ret_cum.diff().fillna(0)
        profit_ts = ret_diff[ret_diff>0]
        loss_ts = ret_diff[ret_diff<0]

        try:
            profit = profit_ts.cumsum().values[-1]
            loss = loss_ts.cumsum().values[-1]

        except:
            profit, loss = 0, 1

        win_times = profit_ts.shape[0]
        lose_times = loss_ts.shape[0]

        annual_sharpe = ret_ts.mean() / ret_ts.std() * np.sqrt(annual_factor)
        annual_vol =  ret_ts.std() * np.sqrt(annual_factor)

        ret_to_vol = total_return / annual_vol
        ret_to_mdd = total_return / mdd
        cagr_to_vol = cagr / annual_vol
        cagr_to_mdd = cagr / mdd

        profit_to_loss = profit/(-loss) #if ~loss==0 else 0

        # max dd cousective periods
        max_dd_period = 0
        current_consecutive = 0

        # Iterate through the Series
        for value in ts_dd:
            if value < 0:
                current_consecutive += 1

                max_dd_period = max(max_dd_period, current_consecutive)
            else:
                current_consecutive = 0

        try: win_rate = win_times / (win_times + lose_times)  # in case no enter
        except: win_rate = 0
        
        perf_dict = {
            'name': name,
            
            'Total_Return(%)': total_return*100,
            'CAGR(%)': cagr*100,
            'Annual_Sharpe': annual_sharpe,
            'Annual_Vol': annual_vol,

            'MDD(%)': -mdd*100,
            'max_dd_period': -max_dd_period,

            'Ret_to_Vol': ret_to_vol,
            'Ret_to_MDD':ret_to_mdd,
            'CAGR_to_Vol': cagr_to_vol,
            'CAGR_to_MDD':cagr_to_mdd,

            'profit_to_loss': profit_to_loss,
            'Win_Rate(%)': win_rate*100,
            # TODO
            #'aveg_holding_period': aveg_holding_period,
            #'Total_Trades': pf.trades.count(),
        }
        return pd.DataFrame([perf_dict])

    def get_perf_table_from_dataframe(
            self, 
            df_ret:pd.DataFrame, 
            annual_factor:float = 252, 
            is_compound:bool = True,
            need_perf_columns:list = ['CAGR(%)', 'Annual_Sharpe', 'MDD(%)', 'max_dd_period', 'profit_to_loss', 'Win_Rate(%)'], 
            is_show:bool = True, 
            is_return:bool = False,
        ):
        """
        Get whole return dataframe Performance Table

        Args:
            df_ret (pd.DataFrame): return time series dataframe
            annual_factor (float): annual periods
            is_compound (bool): if compound return
            need_perf_columns (list): needed performance columns
            is_show (bool): if show performance table
            is_return (bool): if return performance table

        Return:
            pd.DataFrame: Performance Table
        """   
        stats_df = pd.concat([
            self.get_perf_table(
                ret = df_ret[col],
                name = col,
                annual_factor = annual_factor,
                is_compound = is_compound,
            )
        for col in df_ret.columns], axis=0).set_index('name')[need_perf_columns].round(3)
        print(tabulate(stats_df, headers='keys', tablefmt='psql')) if is_show else None
        if is_return:
            return stats_df

    def get_rolling_perf_table(
            self, 
            ret:pd.Series,  lev:float=1, 
            year:int=1, start_year:int=2020, end_year:int=2024, 
            annual_factor:float = 252, 
            is_compound:bool = True,
            need_perf_columns:list = ['CAGR(%)', 'Annual_Sharpe', 'MDD(%)', 'max_dd_period', 'profit_to_loss', 'Win_Rate(%)'], 
            is_show:bool = True, 
            is_return:bool = False,
        ):
        """
        Get Rolling Performance Table

        Args:
            ret (pd.Series): return time series
            lev (float): leverage
            year (int): rolling year(s)
            start_year (int): start year
            end_year (int): end year
            annual_factor (float): annual periods
            is_compound (bool): if compound return
            need_perf_columns (list): needed performance columns
            is_show (bool): if show performance table
            is_return (bool): if return performance table

        Return:
            pd.DataFrame: Rolling Performance Table
        """         
        df = pd.concat([
            self.get_perf_table(
                (ret[str(i):str(i+year-1)])*lev, 
                annual_factor=annual_factor, 
                is_compound=is_compound, 
                name = f'{i}~{i+year-1}'
            ) for i in range(start_year, end_year-year+2)
        ], axis='rows')
        
        stats_df = pd.concat([
            df,
            self.get_perf_table(
                (ret[f"{start_year}-1-1":f"{end_year}-12-31"]*lev), 
                annual_factor=annual_factor, is_compound=is_compound, name='whole'
            )
        ], axis='rows').set_index('name')[need_perf_columns].round(3) 
        print(tabulate(stats_df, headers='keys', tablefmt='psql')) if is_show else None
        if is_return:
            return stats_df


