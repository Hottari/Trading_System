import asyncio
import sys, os

import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.extend([PROJECT_ROOT, '..', '../..']) 
from Data_Loader.data_loader import DataLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None, long_short_ratio_type='global'):
    loader = DataLoader(
        exchange = exchange, 
        symbol_type = symbol_type, 
        start = start,
        end = end,
        timezone = timezone,
        long_short_ratio_type = long_short_ratio_type,
    )
    save_dir_lsr = os.path.join(PROJECT_ROOT, 'data_base', exchange, 'usd', 'long_short_ratio', long_short_ratio_type, freq)
    tasks = [
        loader.do_fetch_update(save_dir=save_dir_lsr, item='long_short_ratio', freq=freq, symbol_li=symbol_li),
    ]
    await asyncio.gather(*tasks)

params = {
    'exchange': 'binance',
    'symbol_type': 'long_short_ratio',
    'timezone': 'UTC',
    'start': '2024-4-1',                # only 30 days data, earily will fail
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
for long_short_ratio_type in ['top_account', 'top_position']: #['global', 'top_account', 'top_position']:
    params['long_short_ratio_type'] = long_short_ratio_type
    asyncio.run(update_data(**params))

