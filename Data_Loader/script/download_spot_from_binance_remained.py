import sys, os
import requests

sys.path.insert(1, f"{os.getcwd()}/../") 
from binance_loader import BinanceLoader

timezone='Asia/Taipei'
binance = BinanceLoader(timezone=timezone)

if __name__ == '__main__':

    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    exchange_info = response.json()
    spot_li1 = [i['symbol'] for i in exchange_info['symbols'] if i['quoteAsset']=='USDT']

    start = '2016-01-01 00:00:00+08:00'
    freq = '30m'
    binance.download_spot_ohlcv(start, freq, spot_li1[227:])



