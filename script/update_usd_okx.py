import asyncio
import sys, os
sys.path.extend(['..', '../..']) 
from Data_Loader.api_connecter.okx_loader import OKXLoader

async def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    ol = OKXLoader(exchange, symbol_type, timezone)
    
    # Create tasks for updating OHLCV and funding rate data
    tasks = [ol.update_ohlcv(start=start, end=end, freq=freq, symbol_li=symbol_li),
             ol.update_funding_rate(start=start, end=end, symbol_li=symbol_li)]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)


params = {
    'exchange': 'okx',
    'symbol_type': 'usd',
    'timezone': 'UTC',
    'start': '2015-1-1',
    'end': None,
    'freq': '15m',
    'symbol_li': [
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "BNB-USDT-SWAP",
        "SOL-USDT-SWAP",
        "XRP-USDT-SWAP",
        "ADA-USDT-SWAP",
        "AVAX-USDT-SWAP",
        "LINK-USDT-SWAP",
        "DOT-USDT-SWAP",
        "TRX-USDT-SWAP"
    ],
}

# Run the async function
asyncio.run(update_data(**params))