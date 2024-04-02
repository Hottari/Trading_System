import asyncio
import sys, os
sys.path.extend(['..', '../..']) 
from Data_Loader.api_connecter.binance_loader import BinanceLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    bl = BinanceLoader(exchange, symbol_type, timezone)
    
    # Create tasks for updating OHLCV and funding rate data
    tasks = [bl.update_ohlcv(start=start, end=end, freq=freq, symbol_li=symbol_li),
             bl.update_funding_rate(start=start, end=end, symbol_li=symbol_li)]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)


params = {
    'exchange': 'binance',
    'symbol_type': 'usd',
    'timezone': 'UTC',
    'start': '2015-1-1',
    'end': None,
    'freq': '15m',
    'symbol_li': [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "TRXUSDT"
    ],
}

# Run the async function
asyncio.run(update_data(**params))