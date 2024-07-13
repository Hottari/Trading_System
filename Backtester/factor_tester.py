import os, sys
import pandas as pd
import numpy as np
import cufflinks as cf
cf.go_offline()

sys.path.extend(['..', '../..'])
from Backtester.backtest import BackTester
bt = BackTester()

from tabulate import tabulate
import plotly.graph_objects as go


class FactorTester():
    def __init__(self):
        pass

    # =================== data processing =================== #
    def do_de_outlier(self, df, n=3, method='median'):
        if method == 'median':
            medi = df.median(axis=1)
            mad_e = df.sub(medi, axis=0).abs().median(axis=1)
            upper = medi + n*mad_e
            lower = medi - n*mad_e
            df_de_outlier = df.clip(lower=lower, upper=upper, axis=0)
        return df_de_outlier


    def get_ret_ls(
            self, 
            rets, fr, factor, 
            slip_rate=0, fee_rate_in=0.0005, fee_rate_out=0.0005,
            signal_delay_periods = 0
        ):
        df = factor.dropna(axis='columns', how='all').dropna(axis='rows', how='all').copy()
        factor_cs_rank = df.rank(axis='columns', method='first')        # 'first' -> 簡易處理避免同名次

        # 向中位數偏移以達到 ls 數量一致
        demean_sign = np.sign(factor_cs_rank.sub(factor_cs_rank.median(axis=1), axis=0))
        weight_ls = demean_sign.div(demean_sign.abs().sum(axis=1), axis=0).fillna(0)

        # ret_ls
        ret_ls = bt.get_ret_after_fric(
            rets = rets, 
            weights = weight_ls, 
            fr = fr,
            slip_rate=slip_rate, fee_rate_in=fee_rate_in, fee_rate_out=fee_rate_out,
            signal_delay_periods = signal_delay_periods,
        ).dropna(axis='columns', how='all').dropna(axis='rows', how='all').sum(axis='columns')

        long_only = demean_sign[demean_sign>0]
        weight_lo = long_only.div(long_only.sum(axis=1), axis=0).fillna(0)
        ret_lo = bt.get_ret_after_fric(
            rets = rets, 
            weights = weight_lo, 
            fr = fr,
            slip_rate=slip_rate, fee_rate_in=fee_rate_in, fee_rate_out=fee_rate_out,
            signal_delay_periods = signal_delay_periods,
        ).dropna(axis='columns', how='all').dropna(axis='rows', how='all').sum(axis='columns')
        return {
            'ret_lo': ret_lo,
            'ret_ls': ret_ls,
            'weight_ls': weight_ls,
            'weight_lo': weight_lo,
        }


    def get_turnover(self, weighting:pd.DataFrame):
        delta_weight = weighting.diff()
        daily_trading_value = delta_weight.abs().sum(axis=1)
        turnover = daily_trading_value.sum() / len(daily_trading_value)
        return turnover


    def plot_alpha_and_LS(self, ret_alpha, ret_ls=None, ret_bm=None, bm_name='btc', factor_name='factor', is_ls=True):
        ret_bm = self.ret_txf if ret_bm is None else ret_bm
        start = max(ret_alpha.index[0], ret_ls.index[0], ret_bm.index[0])
        end = min(ret_alpha.index[-1], ret_ls.index[-1], ret_bm.index[-1])
        x = ret_alpha[start:end].index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=(ret_alpha[start:end]+1).cumprod()-1, mode='lines', name='Alpha', opacity=0.8, line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x, y=(ret_ls[start:end]+1).cumprod()-1, mode='lines', name='LS', line=dict(color='orange'))) if is_ls else None
        fig.add_trace(go.Scatter(x=x, y=(ret_bm[start:end]+1).cumprod()-1, mode='lines', name=bm_name, opacity=0.3, line=dict(color='royalblue')))
        fig.update_layout(
            title=f"Cum Ret ( {factor_name} )", title_x=0.5,
            xaxis_title='datetime', yaxis_title='return', autosize=False, width=900, height=450, 
            shapes=[dict(
                type="line", yref="y", y0=0, y1=0,
                xref="paper", x0=0, x1=1,
                line=dict(color="rgba(0, 0, 0, 0.5)", width=1, dash="dash",
            ))])
        fig.show()



    def do_factor_test(self, factor, fr, rets, need_perf_columns=['CAGR(%)', 'Annual_Sharpe', 'MDD(%)']):
        test_result = self.get_ret_ls(
            rets = rets,
            fr = fr,
            factor = factor,
        )
        ret_ls = test_result['ret_ls']
        ret_alpha = test_result['ret_lo']
        print(f"turnover (long short): {(self.get_turnover(test_result['weight_ls'])*100).round(2)}%")
        print(f"turnover (long only) : {(self.get_turnover(test_result['weight_lo'])*100).round(2)}%")
        print()

        start, end = max(ret_alpha.index[0], ret_ls.index[0]), min(ret_alpha.index[-1], ret_ls.index[-1])
        print(f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
        df_perf = pd.concat([
            bt.get_perf_table(ret=ret_alpha[start:end], name='alpha', is_compound=True),
            bt.get_perf_table(ret=ret_ls[start:end], name='ls', is_compound=True),
            bt.get_perf_table(ret=rets['BTCUSDT'][start:end], name='btc', is_compound=True),
        ], axis='rows').set_index('name')[need_perf_columns].round(3)
        print(tabulate(df_perf, headers='keys', tablefmt='psql'))
        self.plot_alpha_and_LS(ret_alpha=ret_alpha, ret_ls=ret_ls)