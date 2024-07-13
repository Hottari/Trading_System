import time
import sys, os

sys.path.insert(1, os.path.expanduser(f'{os.path.dirname(__file__)}/../../'))
from pybit.unified_trading import HTTP as _HTTP_V5

class BybitHandler:
    def __init__(self, api_key=None, api_secret=None, if_testnet=True):
        self._session_v5 = _HTTP_V5(
            testnet = if_testnet,
            api_key = api_key,
            api_secret = api_secret,
            )
    # =========== Get Info - Wallet =========== #
    
    def get_coin_wallet_balance(self, coins_li:list=None): 
        result = self._session_v5.get_wallet_balance(
            accountType="UNIFIED",
            )['result']['list'][0] 
            
        if coins_li: 
            return [i for i in result['coin'] if i['coin'] in coins_li]
        else: 
            return result['coin']
        

    # =========== Get Info - Market Price =========== #
    def get_last_price(self, symbol):
        result = self._session_v5.get_tickers(category="spot", symbol=symbol)['result']['list'][0]
        return float(result['lastPrice'])


    # =========== Get Info - Open Orders =========== #
    def get_symbol_open_orders(self, symbol=None, orderId=None, limit=50):
        kwargs = {
            'category': 'spot',
            'symbol': symbol, 
            'orderId': orderId,
            'limit': limit,
            }
        result = self._session_v5.get_open_orders(**kwargs)['result']
        orders = {}



        # for details in result['list']:
        #     orders[details['orderId']] = details
        # limit = 20 if 'limit' not in kwargs else kwargs['limit']

        # while len(result['list']) >= limit and result['nextPageCursor'] != '':
        #     kwargs['cursor'] = result['nextPageCursor']
        #     result = self._v5_session.get_active_order(**kwargs)['result']
        #     for details in result['list']:
        #         orders[details['orderId']] = details

        # open_orders = {}
        # for _, details in orders.items():
        #     open_orders[details['orderId']] = Order(
        #         details['orderId'], 
        #         details['symbol'], 
        #         float(details['price']), 
        #         float(details['qty']), 
        #         details['orderStatus'],                           
        #         float(details['cumExecQty']), 
        #         get_buy_sell_inline(details['side'])
        #     )
        return result#orders

    # =========== Get Info - Trading Records =========== #


    # =========== Get Info - Order Records =========== #


    # =========== Make Action - Trade =========== #


    # =========== Make Action - Cancel Order =========== #
