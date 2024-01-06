import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import os



# Plot
def plot_distribution(data:np.ndarray, data_name:str='', bins:int=100):
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



def plot_interactive(df:pd.DataFrame, x:str, y:list, title:str='Interactive_Plot'):
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
