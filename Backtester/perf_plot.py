import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import os




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



    def plot_mdd(df_equity:pd.DataFrame, equity_name:str, separate_date:str=None):
        equity = df_equity.copy()
        fig, x = plt.subplots(figsize = (12,5))

        # Equity
        equity[equity_name].plot(
            label = 'Equity', 
            ax = x, 
            c = 'gray', 
            grid = True
        )

        # New high
        peak_index = equity[equity[equity_name].cummax() == equity[equity_name]].index
        plt.scatter(
            peak_index, 
            equity[equity_name].loc[peak_index],
            c = '#02ff0f', 
            label ='New High')
        if separate_date: plt.axvline(x = pd.to_datetime(separate_date), linestyle ="--", c='blue', alpha = 0.4)

        # Drawdown
        equity['drawdown'] = equity[equity_name] - equity[equity_name].cummax()
        plt.fill_between(equity['drawdown'].index, equity['drawdown'], 0, facecolor='r', label='Drawdown', alpha=0.5)
        
        plt.legend()
        plt.ylabel(f"{equity_name}")
        plt.xlabel('Date')
        plt.title(f'Profit & Drawdown', fontsize=16)
        plt.show()



    def plot_monthly_return(monthly_return_df:pd.DataFrame):
        # Convert to a DataFrame
        df = monthly_return_df

        # Ensure the 'Date' column is a datetime type
        df['Date'] = pd.to_datetime(df['Date'])

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

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0, 
                        norm=TwoSlopeNorm(vmin=pivot_table.min().min(), vcenter=0, vmax=pivot_table.max().max()),
                        fmt=".2f")
        plt.title('Monthly Portfolio Performance Heatmap (Return * 100)')
        plt.show()