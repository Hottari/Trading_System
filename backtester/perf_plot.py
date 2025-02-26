import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from itertools import cycle


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cufflinks as cf
cf.go_offline()

from plotly.subplots import make_subplots



class PerfPlot():

    def __init__(self):
        pass

    # ================== Data Attributes ==================
    def plot_distribution_plotly(self, data: np.ndarray, data_name:str = '', bins:int=20, width=900, height=400):
        """
        Plot data distribution using Plotly's iplot method.

        Args:
            data (array or Series): Data to plot
            data_name (str): Title of the plot
            bins (int): Number of bins for the distribution

        Example:
            plot_distribution(data, 'Volume', bins=100)
        """
        # Convert data to DataFrame
        df = pd.Series(data, name=data_name)
        fig = df.iplot(kind='hist', bins=bins, histnorm='probability density', title=f'Distribution {data_name}', 
                xTitle=data_name, yTitle='Density', colors='skyblue', asFigure=True)
        fig.update_traces(
            marker=dict(
                color = 'rgba(75, 192, 192, 0.6)',      # Fill color
                line=dict(
                    color = 'rgba(0, 0, 0, 1)',         # Border color
                    width=1,
                )
            )
        )
        fig.update_layout(width=width, height=height, bargap=0.1)
        fig.show()

    def plot_distribution(self, data:np.ndarray, data_name:str='', bins:int=100):
        """
        plot data distribution

        Args
            data (array or Series): 要畫的 data
            data_name: 圖標題
            bins (int): 分布組數

        Example:
            plot_distribution(data, 'Volumne', bins=100)  
        """
        plt.hist(data, bins, density=True, color='skyblue', edgecolor='black', alpha=0.7)

        # Add labels and title
        plt.xlabel(data_name)
        plt.ylabel('Density')
        plt.title(f'Distribution of {data_name}')

        # Show the plot
        plt.show()


    # ================== Performance Metrics ==================
    def plot_interactive_line(self, df:pd.DataFrame, x:str, y:list, title:str='Interactive_Plot'):
        """
        plot interactive figure

        Args
            df (pd.DataFrame): 包含要畫的整個 DateDrame
            x (str os list): x軸, 通常為 datetime index
            y (list): 要畫的 y名稱之 list
            title (str): 圖標題

        Example:
            interactive_plot(df=df, x='index', y=name_li, title=f"Index_{data_freq}_{year}")  
        """
        # Create Plotly figure using graph_objects
        fig = go.Figure()

        # Add lines to the figure with initial visibility set to 'legendonly'
        for column in y:
            fig.add_trace(go.Scatter(x=df[x], y=df[column], mode='lines', name=column, visible='legendonly'))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            xaxis=dict(rangeslider=dict(visible=True)),
        )

        # Save the interactive plot to an HTML file
        html_file_path = f"{title}.html"
        pyo.plot(fig, filename=html_file_path, auto_open=False)

        # Print the path to the HTML file
        print(f"Interactive plot saved to: {os.path.join(os.getcwd(), html_file_path)}")


    def plot_mdd_plotly(
        self, 
        df, data_name, 
        title=None, is_ret=True, 
        is_save=False, separate_date=None,
        height = 500, width = 1000,
    ):
        if is_ret:
            df['drawdown'] = (df[data_name]+1) / (np.maximum(0, df[data_name].cummax())+1) - 1
        else:
            df['drawdown'] = df[data_name] - df[data_name].cummax()    
        peak_index = df[df[data_name].cummax() == df[data_name]].index
        peak_values = df[data_name].loc[peak_index]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Equity', 'Drawdown'), vertical_spacing=0.1)
        # Add traces
        fig.add_trace(go.Scatter(x=df.index, y=df[data_name], name='Equity', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=peak_index, y=peak_values, mode='markers', name='New High', marker=dict(color='#02ff0f')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='Drawdown', fill='tozeroy', fillcolor='rgba(255,0,0,0.3)', marker=dict(color='rgba(0,0,0,0)')), row=2, col=1)
        # Add vertical line
        if separate_date!=None:
            fig.add_shape(
                type = "line",
                x0 = pd.to_datetime(separate_date).strftime('%Y-%m-%d'), y0=0,
                x1 = pd.to_datetime(separate_date).strftime('%Y-%m-%d'), y1=1,
                yref = 'paper',                                     # refers to the entire plotting area
                line = dict(color=f"rgba(0, 32, 91, {0.5})", width=2, dash="dash")    # or dashdot
            )          
        # Update layout
        fig.update_layout(
            height = height, width = width, 
            title_text = title,
            title_font = dict(size = 18)
        )
        fig.show()
        pyo.plot(fig, filename=title, auto_open=False) if is_save else None


    def plot_mdd(self, df_equity: pd.DataFrame, equity_name: str, separate_date: str = None, is_ret: bool = True):
        equity = df_equity.copy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

        # Remove space between subplots
        plt.subplots_adjust(hspace=0)

        # Plotting the equity curve
        ax1.plot(equity.index, equity[equity_name], label='Equity', c='gray')
        ax1.grid(True)

        # Plotting new highs
        peak_index = equity[equity[equity_name].cummax() == equity[equity_name]].index
        ax1.scatter(peak_index, equity[equity_name].loc[peak_index], c='#02ff0f', label='New High')

        # If a separation date is provided
        if separate_date:
            ax1.axvline(x=pd.to_datetime(separate_date), linestyle="--", c='blue', alpha=0.4)
            ax2.axvline(x=pd.to_datetime(separate_date), linestyle="--", c='blue', alpha=0.4)

        # Calculating drawdown
        if is_ret:
            equity['drawdown'] = (equity[equity_name]+1) / (np.maximum(0, equity[equity_name].cummax())+1) - 1
        else:
            equity['drawdown'] = equity[equity_name] - equity[equity_name].cummax()

        # Plotting drawdown
        ax2.fill_between(equity.index, equity['drawdown'], 0, facecolor='r', label='Drawdown', alpha=0.5)

        # Set limits for the y-axis
        y_min = equity['drawdown'].min()
        ax2.set_ylim(y_min, 0)

        # Adding grid to the lower graph
        ax2.grid(True)

        # Adding legend and labels
        fig.legend(loc="upper right")
        ax1.set_ylabel(f"{equity_name}")
        ax2.set_xlabel('Date')
        plt.xticks(rotation=45)
        plt.suptitle(f'Profit & Drawdown of {equity_name}', fontsize=16)
        plt.show()



    def plot_monthly_ret(self, monthly_return_df):
        df = monthly_return_df
        # Extract years and months
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        # Pivot the table for heatmap
        pivot_table = df.pivot(index="Month", columns="Year", values="Return")
        pivot_table *= 100  # Multiplying the returns by 100
        # Reorder pivot_table index to match the calendar months orde
        months_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        pivot_table = pivot_table.reindex(months_order)

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            data = pivot_table,
            norm = TwoSlopeNorm(vmin=pivot_table.min().min(), vcenter=0, vmax=pivot_table.max().max()),
            fmt = ".2f", 
            annot=True,
            annot_kws = {"size": 8},
            ax = ax,  # Ensure ax is specified for the heatmap
            cmap = 'RdYlGn',
        )
        ax.grid(False)  # Remove grid lines (if any)

        plt.title('Portfolio Monthly Return(%)', fontsize=18, pad=20)  # Increase pad value to move the title higher
        plt.xticks(rotation=45)
        plt.show()


    # ================== Real Trading Tracker ==================
    def plot_track_result_plotly(self, lev, df_bal_real, portfolio_name, ret_bt, start, end, timeframe='8h'):
        df_ret = pd.DataFrame({'backtest':ret_bt[start:end]})
        df_ret['real'] = ((df_bal_real[df_bal_real['portfolio'] == portfolio_name]['return'][start:]+1).resample(timeframe).prod()-1).shift(1)
        df_ret = df_ret.iloc[1:]
        df_ret["Real-Backtest"] = df_ret['real']-df_ret['backtest']*lev
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_ret.index, y=(1+df_ret['backtest']*lev).cumprod()-1, mode='lines+markers', name='backtest', line=dict(color='blue', dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_ret.index, y=(1+df_ret['real']).cumprod()-1, mode='lines+markers', name='real', line=dict(color='orange')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_ret.index, y=df_ret["Real-Backtest"], mode='markers', name='real-backtest', marker=dict(color='green', size=10, opacity=0.5)), secondary_y=True)

        # Get the minimum and maximum for each dataset
        y1 = (1+df_ret['backtest']*lev).cumprod()-1
        y2 = df_ret["Real-Backtest"]
        max_y1 = y1.max()
        min_y1 = y1.min()
        max_y2 = y2.max()
        min_y2 = y2.min()
        adj_scale_y1 = (max_y1 - min_y1) / 10

        # Calculate the scale factor between the two datasets
        scale = (max_y1 - min_y1) / (max_y2 - min_y2)
        fig.update_yaxes(title_text="Cumulative Returns", range=[min_y1-adj_scale_y1, max_y1+adj_scale_y1], secondary_y=False)
        fig.update_yaxes(title_text="Real-Backtest", range=[(min_y1-adj_scale_y1) / scale, (max_y1+adj_scale_y1) / scale], secondary_y=True)

        # Add horizontal line at y=0 on second y-axis
        fig.add_shape(type="line", line=dict(dash='dash'), x0=df_ret.index.min(), x1=df_ret.index.max(), y0=0, y1=0, secondary_y=True)
        fig.update_layout(height=450, width=900, title_text=f"Track Result ( {portfolio_name} )")
        fig.show()

    
    def plot_river_chart(self, df_data, figure_width=1200, figure_height=600):
        df = df_data.copy()
        # Select a Plotly Express built-in color sequence
        color_palette = px.colors.qualitative.Set1  # Example color palette
        color_cycle = cycle(color_palette)  # Create a cycle iterator for the color palette
        # Initialize the figure with specified layout dimensions
        fig = go.Figure()
        # Add each asset as a trace with a unique color from the cycle
        for item in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[item],
                hoverinfo='x+y',
                mode='lines',
                line=dict(width=0.5, color=next(color_cycle)),  # Get the next color from the cycle
                stackgroup='one',     # Creates the stacking effect
                groupnorm='percent',  # Normalize to percentage for full area coverage
                name=item
            ))
        # Update the layout with axis labels, title settings, and figure dimensions
        fig.update_layout(
            showlegend=True,
            yaxis=dict(
                title='Asset Weight',  # Y-axis label
                type='linear',
                range=[1, 100],
                ticksuffix='%'
            ),
            title=dict(
                text="River Chart of Asset Weighting Over Time",
                x = 0.5,  # Center the title
                xanchor = 'center',  # Use the middle of the title for centering
                font=dict(  # Customize font settings
                    size=24  # Bigger font size for the title
                )
            ),
            width = figure_width,
            height = figure_height
        )
        fig.show()
