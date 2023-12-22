import sys, os

sys.path.insert(1, f"{os.getcwd()}/../") 
from binance_loader import BinanceLoader

timezone='Asia/Taipei'
binance = BinanceLoader(timezone=timezone)

if __name__ == '__main__':

    start = '2016-01-01 00:00:00+08:00'
    freq = '30m'
    binance.download_spot_ohlcv(start, freq)

