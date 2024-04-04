import asyncio
import sys, os
sys.path.extend(['..', '../..']) 
from Data_Loader.api_connecter.okx_loader import OKXLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    ol = OKXLoader(
        exchange = exchange, 
        symbol_type = symbol_type, 
        timezone = timezone
    )
    
    # Create tasks for updating OHLCV and funding rate data
    tasks = [ol.update_ohlcv(start=start, end=end, freq=freq, symbol_li=symbol_li),]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)


params = {
    'exchange': 'okx',
    'symbol_type': 'spot',
    'timezone': 'UTC',
    'start': '2015-1-1',
    'end': None,
    'freq': '4H',
    'symbol_li': [
        "BTC-USDT",
        "ETH-USDT",
        "BNB-USDT",
        "SOL-USDT",
        "XRP-USDT",
        "ADA-USDT",
        "AVAX-USDT",
        "LINK-USDT",
        "DOT-USDT",
        "TRX-USDT"
    ],
}

# Run the async function
asyncio.run(update_data(**params))