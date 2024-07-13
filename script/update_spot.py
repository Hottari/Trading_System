import asyncio
import sys, os

import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.extend([PROJECT_ROOT, '..', '../..']) 
from Data_Loader.data_loader import DataLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    loader = DataLoader(
        exchange = exchange, 
        symbol_type = symbol_type, 
        start = start,
        end = end,
        timezone = timezone,
    )
    save_dir_ohlcv = os.path.join(PROJECT_ROOT, 'data_base', exchange, symbol_type, 'ohlcv', freq)

    tasks = [
        loader.do_fetch_update(save_dir=save_dir_ohlcv, item='ohlcv', symbol_li=symbol_li, freq=freq),
    ]
    # Run tasks concurrently
    await asyncio.gather(*tasks)


params = {
    'exchange': 'binance',
    'symbol_type': 'spot',
    'timezone': 'UTC',
    'start': '2010-1-1',
    'end': None,
    'freq': '1h',
    # 'symbol_li': [
    #     "BTCUSDT",
    #     "ETHUSDT",
    #     "BNBUSDT",
    #     "SOLUSDT",
    #     "XRPUSDT",
    #     "ADAUSDT",
    #     "AVAXUSDT",
    #     "LINKUSDT",
    #     "DOTUSDT",
    #     "TRXUSDT"
    # ],
}

asyncio.run(update_data(**params))

