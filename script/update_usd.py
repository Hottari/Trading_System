import asyncio
import sys, os

import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.extend([PROJECT_ROOT, '..', '../..']) 
from data_loader.data_loader import DataLoader

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(PROJECT_ROOT, 'config', 'file_path_config.ini'))
SAVE_ROOT = config.get('path', 'crypto_data_path')

# async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    loader = DataLoader(
        exchange = exchange, 
        symbol_type = symbol_type, 
        start = start,
        end = end,
        timezone = timezone,
    )
    save_dir_ohlcv = os.path.join(SAVE_ROOT, exchange, symbol_type, 'ohlcv', freq)
    save_dir_fr = os.path.join(SAVE_ROOT, exchange, symbol_type, 'funding_rate')
    # tasks = [
    #     loader.do_fetch_update(save_dir=save_dir_ohlcv, item='ohlcv', symbol_li=symbol_li, freq=freq),
    #     loader.do_fetch_update(save_dir=save_dir_fr, item='funding_rate', symbol_li=symbol_li),
    # ]
    # # Run tasks concurrently
    # await asyncio.gather(*tasks)
    loader.do_fetch_update(save_dir=save_dir_ohlcv, item='ohlcv', symbol_li=symbol_li, freq=freq)
    loader.do_fetch_update(save_dir=save_dir_fr, item='funding_rate', symbol_li=symbol_li)

params = {
    'exchange': 'binance',
    'symbol_type': 'usd',
    'timezone': 'UTC',
    'start': '2010-1-1',
    'end': None,
    'freq': '5m',
    'symbol_li': [
        # "BTCUSDT",
        # "ETHUSDT",
        # "BNBUSDT",
        # "SOLUSDT",
        # "XRPUSDT",
        # "ADAUSDT",
        # "AVAXUSDT",
        # "LINKUSDT",
        # "DOTUSDT",
        # "TRXUSDT",
    ],
}
# asyncio.run(update_data(**params))
update_data(**params)
