import sys, os
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..',)
sys.path.insert(1, PROJECT_ROOT) 

from Data_Loader.api_connecter.binance.binance_loader import BinanceLoader

def update_data(exchange, symbol_type, timezone, start, end, freq, symbol_li=None):
    bl = BinanceLoader(exchange, symbol_type, timezone)
    
    bl.update_ohlcv(start=start, end=end, freq=freq, symbol_li=symbol_li)
    bl.update_funding_rate(start=start, end=end, symbol_li=symbol_li)

if __name__ == '__main__':
    params = {
        'exchange': 'binance',
        'symbol_type': 'usd',
        'timezone': 'UTC',
        'start': '2015-1-1',
        'end': None,
        'freq': '1m',
        #'symbol_li': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT'],
    }

    update_data(**params)