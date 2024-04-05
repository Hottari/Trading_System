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


class PerfPlot():

    def __init__(self):
        pass

    # Plot
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
        plt.suptitle(f'Profit & Drawdown', fontsize=16)
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
            # xaxis=dict(
            #     title='Date',  # X-axis label
            #     type='date'
            # ),
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
