import asyncio
import sys, os
sys.path.extend(['..', '../..']) 
from Data_Loader.api_connecter.binance_loader import BinanceLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    bl_global = BinanceLoader(exchange, symbol_type, 'global', timezone)
    bl_top_account = BinanceLoader(exchange, symbol_type, 'top_account', timezone)
    bl_top_position = BinanceLoader(exchange, symbol_type, 'top_position', timezone)
    
    # Create tasks for updating OHLCV and funding rate data
    tasks = [
        bl_global.update_long_short_ratio(start=start, end=end, freq=freq, symbol_li=symbol_li),
        bl_top_account.update_long_short_ratio(start=start, end=end, freq=freq, symbol_li=symbol_li),
        bl_top_position.update_long_short_ratio(start=start, end=end, freq=freq, symbol_li=symbol_li),
    ]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)


params = {
    'exchange': 'binance',
    'symbol_type': 'usd',
    'timezone': 'UTC',
    'start': '2024-3-4',                    # only 30days data (earlier will fail)
    'end': None,
    'freq': '15m',
    'symbol_li': [
        "BTCUSDT",
    ],
}

# Run the async function
asyncio.run(update_data(**params))

