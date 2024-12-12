import MetaTrader5 as mt5
from stable_baselines3 import PPO
import time
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
import numpy as np
from datetime import datetime, timedelta
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv
import logging
import logging.config
import yaml

# Load logging configuration
with open("config/logging.yaml", "r") as file:
    logging_config = yaml.safe_load(file)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger("live_trade")

# Khởi động MT5
if not mt5.initialize():
    print("Lỗi khi khởi động MetaTrader5")
    mt5.shutdown()

class LiveTradeMT5:
    def __init__(self, model,symbol,balance,size_lot,leveral):
        self.model = model
        self.symbol = symbol
        self.lot_size = size_lot
        self.pip=0
        self.loss_streak=0
        self.win_streak=0
        self.tp=15
        self.sl=15
        self.spread=0
        self.balance=balance
        self.leveral=leveral
        self.position=0
        self.entry_price=0
        self.current_price=0
        self.equity=balance
        self.win=0
        self.is_close=0
        self.obs_columns = ['Datetime_entry','Mua/Ban','entry_price',
        'ema10', 'ema20', 'ema50', 'rsi', 'macd_value', 'macd_signal', 'macd_hist',
        'volume', 'adx','c_DI','t_DI','Lot','Profit','win'
    ]
        self.obs_entry = None
        self.model_path=r"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\models\ppo_trading_xauusd.zip"
        self.observation_buffer = []
        logger.info(f"Initialized LiveTradeMT5 for symbol: {symbol}")
        logger.info(f"Fetched account balance: {balance}")


    @staticmethod
    def get_balance():
        account_info = mt5.account_info()
        if account_info is None:
            print("Lỗi khi lấy thông tin tài khoản.")
            return None
        return account_info.balance
    
    def adjust_risk_and_profit_limits(self):
            if self.loss_streak >= 3:  # Nếu có chuỗi thua lỗ
                self.lot_size=max(0.01,self.lot_size-0.01)
            elif self.win_streak >= 1:  # Nếu có chuỗi thắng
               self.lot_size+=0.03


    def calculate_pip(self,entry_price,current_price):
        print(f"entry_price={entry_price}")
        print(f"current_price={current_price}")
        return self.get_current_profit()/(self.lot_size*10)

    @staticmethod
    def get_historical_data(symbol):
        candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)  # Lấy 100 nến để tính các chỉ báo
        if candles is None or len(candles) == 0:
            
            logger.error(f"Failed to fetch historical data for {symbol}")
            return None
        df = pd.DataFrame(candles)
        return df

    @staticmethod
    def get_current_price(symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to fetch current price for {symbol}")
            return None
        return tick.bid
    
    def get_current_profit(self):
        # Lấy vị thế hiện tại theo symbol
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return 0  # Nếu không có vị thế mở, trả về lợi nhuận bằng 0
        # Giả sử chỉ có một vị thế mở, bạn có thể thay đổi để lấy tổng lợi nhuận của tất cả vị thế
        return positions[0].profit


    @staticmethod
    def calculate_indicators(df):
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        ema10 = EMAIndicator(close=df['close'], window=10).ema_indicator()
        ema20 = EMAIndicator(close=df['close'], window=20).ema_indicator()
        ema50 = EMAIndicator(close=df['close'], window=50).ema_indicator()
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        macd_value = macd.macd()
        macd_signal = macd.macd_signal()
        macd_histogram = macd.macd_diff()
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        adx = adx_indicator.adx()
        pos_di = adx_indicator.adx_pos()
        neg_di = adx_indicator.adx_neg()
        
        return {
            "RSI": rsi.iloc[-1],
            "EMA10": ema10.iloc[-1],
            "EMA20": ema20.iloc[-1],
            "EMA50": ema50.iloc[-1],
            "MACD": {"value": macd_value.iloc[-1], "signal": macd_signal.iloc[-1], "histogram": macd_histogram.iloc[-1]},
            "ADX": adx.iloc[-1],
            "+DI": pos_di.iloc[-1],
            "-DI": neg_di.iloc[-1]
        }
    def save_trade_history(self):
        """Lưu lịch sử giao dịch vào file CSV."""
        obs_entry_dict = {name: value for name, value in zip(self.obs_columns, self.obs_entry)}
        
        # Đường dẫn thư mục và file CSV
        directory = r'C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\results\trade_history\real_time'
        if self.symbol == "XAUUSDm":
            file_path = os.path.join(directory, 'trade_history_xau.csv')
        elif self.symbol == "BTCUSDm":
            file_path = os.path.join(directory, 'trade_history_btc.csv')
        elif self.symbol == "USOILm":
            file_path = os.path.join(directory, 'trade_history_usoil.csv')
        else:
            raise ValueError(f"Symbol {self.symbol} không được hỗ trợ!")

        # Kiểm tra xem thư mục đã tồn tại chưa, nếu chưa thì tạo mới
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Kiểm tra xem file đã tồn tại chưa
        file_exists = os.path.isfile(file_path)

        # Chuyển đổi từ điển thành DataFrame
        df = pd.DataFrame([obs_entry_dict])

        # Ghi vào file CSV, nếu file đã tồn tại thì chỉ thêm dữ liệu, không ghi lại header
        df.to_csv(file_path, mode='a', header=not file_exists, index=False)
        logger.info(f"Saved trade history to {file_path}")

    def get_most_recent_deal(self,symbol):
        # Thời gian hiện tại và khoảng thời gian lấy dữ liệu
        to_date = datetime.now()
        from_date = to_date - timedelta(seconds=3)


        # Lấy lịch sử giao dịch trong khoảng thời gian đã chỉ định
        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None or len(deals) == 0:
            print(f"No deals found in the last {3} seconds.")
            return 0

        # Lọc giao dịch theo symbol cụ thể
        symbol_deals = [deal for deal in deals if deal.symbol == symbol]

        if symbol_deals:
            # Lấy giao dịch gần nhất
            recent_deal = max(symbol_deals, key=lambda d: d.time)
            print(f"Most recent deal for {symbol}:")
            print(f"usd: {recent_deal.profit}, Time: {datetime.fromtimestamp(recent_deal.time)}, Type: {recent_deal.type}, Volume: {recent_deal.volume}, Price: {recent_deal.price}")
            return recent_deal.profit/(self.lot_size*3)
            
        else:
            print(f"No deals found for symbol {symbol} in the  {10} seconds.")
            return 0


    @staticmethod
    def calculate_fibonacci(high, low):
        diff = high - low
        return {
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff
        }
    

    def update_real_time_indicators(self):
        df = self.get_historical_data(self.symbol)
        current_price = self.get_current_price(self.symbol)
        
        observation = np.array([])
        if df is not None and current_price is not None and self.balance !=0:
            df.loc[df.index[-1], 'close'] = current_price
            indicators = self.calculate_indicators(df)
            high = df['high'].rolling(window=50).max().iloc[-1]
            low = df['low'].rolling(window=50).min().iloc[-1]
            fibonacci_levels = self.calculate_fibonacci(high, low)
            observation = np.append(observation, [
                indicators['EMA10'], indicators['EMA20'],indicators["EMA50"],indicators['RSI'], indicators['MACD']['value'], indicators['MACD']['signal'],
                indicators['MACD']['histogram'],df['tick_volume'].iloc[-1], indicators['ADX'],indicators["+DI"],indicators["-DI"],current_price
            ])
            
           
 
            
        return observation, {}

    def place_order(self, action,obs):
        tick = mt5.symbol_info_tick(self.symbol)
        vi_the=None
        if action == 0 :  # Mua
            order_type = mt5.ORDER_TYPE_BUY
            self.spread=tick.bid-tick.ask
            self.entry_price=tick.ask
            self.position=1
            logger.info(f"Placing buy order for {self.symbol} at price: {self.entry_price}")
            vi_the="Mua"
        elif action == 1 :  # Bán
            order_type = mt5.ORDER_TYPE_SELL
            self.spread=tick.bid-tick.ask
            self.entry_price=tick.bid
            self.position=-1
            logger.info(f"Placing sell order for {self.symbol} at price: {self.entry_price}")
            vi_the="Ban"
        self.obs_entry=np.append(self.obs_entry,vi_the)
        self.obs_entry=np.append(self.obs_entry,self.entry_price)
        self.obs_entry=np.append(self.obs_entry,obs)
        price_stop_loss=0
        price_tp=0
        if(self.symbol=="BTCUSDm"):
            if( order_type == mt5.ORDER_TYPE_BUY):
               price_stop_loss=self.entry_price-self.sl*10 
               price_tp=self.tp*10 +self.entry_price
            else:
               price_tp=self.entry_price-self.tp*10 
               price_stop_loss=self.sl*10 +self.entry_price
        elif(self.symbol=="XAUUSDm"):
             if( order_type == mt5.ORDER_TYPE_BUY):
               price_stop_loss=self.entry_price-self.sl/10 
               price_tp=self.tp/10 +self.entry_price
             else:
               price_tp=self.entry_price-self.tp/10 
               price_stop_loss=self.sl/10 +self.entry_price

        price = mt5.symbol_info_tick(self.symbol).ask if action == 0 else mt5.symbol_info_tick(self.symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl":price_stop_loss,
            "tp":price_tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "Live Trade MT5",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to place order. Error code: {result.retcode}")

    def close_order(self):
        positions = mt5.positions_get(symbol=self.symbol)
        for position in positions:
            price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Lỗi khi đóng lệnh: {result.retcode}")
        feature_columns = [
            "ema10", "ema20", "ema50", "rsi",
            "macd", "macd_signal", "macd_hist",
            "volume", "ADX", "c_DI", "t_DI","close","datetime"
        ]
        self.pip=self.get_most_recent_deal(self.symbol)
        buffer_df = pd.DataFrame(self.observation_buffer, columns=feature_columns)
        # Tạo môi trường với buffer để huấn luyện lại mô hình
        env = DummyVecEnv([lambda: TradingEnv(buffer_df)])
        self.model.set_env(env)

        # Huấn luyện lại mô hình
        self.model.learn(total_timesteps=len(self.observation_buffer))
        self.model.save(self.model_path)
        self.model = PPO.load(self.model_path)

        # Xóa buffer sau khi cập nhật
        self.observation_buffer = []
        
        profit=0
        if self.pip < -10: 
            self.win=-1        
        elif self.pip>5:    
            self.win=1
        profit=self.pip*(self.lot_size*10)
        self.balance+=profit
        self.equity=self.balance
        self.obs_entry=np.append(self.obs_entry,self.lot_size)
        self.obs_entry=np.append(self.obs_entry,profit)
        self.obs_entry=np.append(self.obs_entry,self.win)
        self.save_trade_history()
           
           
        

    def round_to_two_decimal_places(number):
        return round(number, 2)

    def is_runBot(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if(self.symbol=="BTCUSDm"):
            self.leveral=400
        contract_value_ask = tick.ask * self.lot_size
        contract_value_bid = tick.bid * self.lot_size 
        margin_bid=self.round_to_two_decimal_places(contract_value_bid/self.leveral)
        margin_ask=self.round_to_two_decimal_places(contract_value_ask/self.leveral)
        margin_level_bid=self.equity/margin_bid
        margin__level_ask=self.equity/margin_ask
        if(self.balance<margin_ask or self.balance<margin_bid):return 0
        elif(margin__level_ask<1 or margin_level_bid<1):return 0.5
        else: return 1
              


    def run(self):
        tick = mt5.symbol_info_tick(self.symbol)
        logger.info(f"Starting live trading for {self.symbol}")
        while self.equity!=0 or (positions !=0 and self.is_runBot())!=0 :
            current_datetime=datetime.now()
            formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            obs, _ = self.update_real_time_indicators()
            obs_tmp=np.append(obs,formatted_time)
            self.observation_buffer.append(obs_tmp)
            self.model = PPO.load(self.model_path)
            action, _ = self.model.predict(obs)
            print(f'lot_size={self.lot_size}')
            print(action)
            self.current_price=tick.bid
            if(self.position!=0):
                profit=self.pip*(self.lot_size*10)
                self.equity=self.balance+profit
            if(self.position==1):
                self.current_price=tick.ask
            elif(self.position==-1):
                self.current_price=tick.bid
            else:
                self.entry_price=0
            self.pip=0
            self.win=0
            if(self.position!=0):
                self.pip=self.calculate_pip(self.entry_price,self.current_price)
                print(f'pip={self.pip}')        
            if  (self.position !=0  and  self.get_most_recent_deal(self.symbol)!=0   )  :  # Đóng lệnh         
                 if self.pip < 0:
                    self.loss_streak += 1  # Tăng chuỗi thua lỗ
                    self.win_streak = 0  # Reset chuỗi thắng
                 elif self.pip>0:
                    self.win_streak += 1  # Tăng chuỗi thắng
                    self.loss_streak = 0  # Reset chuỗi thua lỗ
                 self.close_order()
                 print("hello")
                 self.position=0
                 self.pip=0
                 self.entry_price=0
                 

            elif action == 2:  # Giữ lệnh, không vào lệnh mới 
                pass
            else:
                positions = mt5.positions_get(symbol=self.symbol)
                if not positions and self.position==0:
                    self.obs_entry=np.array([])
                    self.obs_entry=np.append(self.obs_entry,formatted_time)
                    self.place_order(action,obs)
            
            time.sleep(1)
        return 0

# Tải mô hình PPO đã huấn luyện
account_info = mt5.account_info()

def get_balance():
        account_info = mt5.account_info()
        if account_info is None:
            print("Lỗi khi lấy thông tin tài khoản.")
            return None
        return account_info.balance
balance=get_balance()
leverage=account_info.leverage
model = PPO.load(r"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\models\ppo_trading_xauusd.zip")

# Khởi tạo live trade
live_trader = LiveTradeMT5(model,"XAUUSDm",balance,0.01,leverage)
live_trader2=LiveTradeMT5(model,"BTCUSDm",balance,0.01,400)
import threading

# Giả sử bạn đã khởi tạo các đối tượng live_trader và live_trader2

# Tạo các hàm để chạy từng live trader
def run_live_trader1():
    live_trader.run()

def run_live_trader2():
    live_trader2.run()

# Tạo và khởi động các luồng
thread1 = threading.Thread(target=run_live_trader1)
thread2 = threading.Thread(target=run_live_trader2)

# Bắt đầu các luồng
thread1.start()
thread2.start()

# Tùy chọn: Đợi cho đến khi các luồng hoàn thành
thread1.join()
thread2.join()
