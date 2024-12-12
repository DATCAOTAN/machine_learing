import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, render_mode=None):
        super(TradingEnv, self).__init__()

        # Dữ liệu và các biến trạng thái
        self.data = data
        self.balance = 100  # Số dư ban đầu
        self.current_step = 0
        self.entry_price = 0
        self.position = 0
        self.loss_streak = 0
        self.win_streak = 0
        self.lot_size = 0.01
        self.pip = 0
        self.loss_pip = 0
        self.win_pip = 0
        self.spread = 2
        self.render_mode = render_mode

        # Không gian hành động (Mua, Bán, Giữ)
        self.action_space = spaces.Discrete(3)
        # Không gian quan sát với 11 chỉ báo kỹ thuật
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)

        # Cột quan sát
        self.obs_columns = ['Datetime_entry', "Mua/Ban", 'entry_price',
                            'ema10', 'ema20', 'ema50', 'rsi', 'macd_value', 'macd_signal', 'macd_hist',
                            'volume', 'ADX', 'c_DI', 't_DI', 'Lot', 'Profit', 'win']
        self.obs_entry = None
        self.win = 0

    def adjust_risk_and_profit_limits(self):
        """Điều chỉnh giới hạn rủi ro và lợi nhuận dựa trên chuỗi thắng/thua"""
        if self.loss_streak >= 10:  # Nếu có chuỗi thua lỗ
            self.lot_size = max(0.01, self.lot_size - 0.01)
        elif self.win_streak >= 10:  # Nếu có chuỗi thắng
            self.lot_size += 0.01

    def calculate_pip(self, entry_price, current_price):
        entry_price = float(entry_price)
        current_price = float(current_price)
        """Tính toán PIP cho mỗi giao dịch"""
        if self.position == 1:
            pip = (current_price - entry_price - self.spread) * 10
        elif self.position == -1:
            pip = (entry_price - current_price - self.spread) * 10
        return pip

    def save_trade_history(self):
        """Lưu lịch sử giao dịch vào file CSV"""
        obs_entry_dict = {name: value for name, value in zip(self.obs_columns, self.obs_entry)}

        # Đường dẫn thư mục và file CSV
        directory = r"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\results\backtest"
        file_path = os.path.join(directory, 'trade_history_xau.csv')

        # Kiểm tra xem thư mục đã tồn tại chưa, nếu chưa thì tạo mới
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Kiểm tra xem file đã tồn tại chưa để chỉ thêm dữ liệu vào cuối file
        file_exists = os.path.isfile(file_path)

        # Chuyển đổi từ điển thành DataFrame và lưu vào file CSV
        df = pd.DataFrame([obs_entry_dict])
        df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        profit=0
        current_price = self.data.iloc[self.current_step]['close']
        datatime= self.data.iloc[self.current_step]['datetime']
        reward = 0  # Phần thưởng ban đầu
        truncated = False
        self.pip=0
        self.win=0
        self.is_close=0
        if(self.position !=0 ):      
            self.pip=self.calculate_pip(self.entry_price,current_price)
        else:
            self.entry_price=0      
        # Xử lý các hành động

        if (self.position !=0  and  ( self.pip>=15 or self.pip<=-15 or done==True) ) :  # Đóng vị thế
           
            if self.pip < 0:
                self.loss_streak += 1  # Tăng chuỗi thua lỗ
                self.win_streak = 0  # Reset chuỗi thắng    
                reward-=1.5
                self.win=-1
                profit=-15
            elif self.pip>0:
                self.win_streak += 1  # Tăng chuỗi thắng
                self.loss_streak = 0  # Reset chuỗi thua l
                profit=10
                reward+=1.5
                self.win=1
            if self.obs_entry is not None:
               self.obs_entry=np.append(self.obs_entry,self.lot_size)
               self.obs_entry=np.append(self.obs_entry,profit)
               self.obs_entry=np.append(self.obs_entry,self.win)
        # Gọi hàm ghi lịch sử giao dịch
            self.save_trade_history()
            self.position = 0 
            self.is_close=1
            self.balance += profit
            # if(done==True):
            #      print("done=TRUE")
            #      print(f'action={action} balance={self.balance}')
            #      print(f'win={self.win_pip} loss={self.loss_pip}')

        elif action == 0 and self.position ==0:  # Mua
            self.position = 1
            self.dem=0
            self.entry_price = current_price
            self.dem_buy+=1
            self.dem_sell=0
            self.obs_entry=np.array([])
            self.obs_entry=np.append(self.obs_entry,datatime)
            self.obs_entry=np.append(self.obs_entry,"Mua")
            self.obs_entry=np.append(self.obs_entry,self.entry_price)
            self.obs_entry=np.append(self.obs_entry,self._next_observation()) 
        elif action == 1 and self.position ==0:  # Bán
            self.position = -1
            self.dem=0
            self.dem_sell+=1
            self.dem_buy=0
            self.entry_price = current_price
            self.obs_entry=np.array([])
            self.obs_entry=np.append(self.obs_entry,datatime)
            self.obs_entry=np.append(self.obs_entry,"Ban")
            self.obs_entry=np.append(self.obs_entry,self.entry_price)
            self.obs_entry=np.append(self.obs_entry,self._next_observation())
        elif(action==2 and self.position==0):
            self.dem+=1
            reward-=1
        
        elif(action==2 and self.position!=0):
            reward+=0.01
        
        
        
        # if((action==0 and self.position==-1 ) or (action==1 and self.position==1) ):
        #     reward-=1
        return self._next_observation(), reward, done, truncated, {'profit':profit}

    def _next_observation(self):
        # Giả sử các yếu tố trong observation là các chỉ báo kỹ thuật
        observation = np.array([])

        # Tính các chỉ báo từ dữ liệu và thêm vào observation
        # Thí dụ:
        observation = np.append(observation, self.data.iloc[self.current_step]['ema10'])
        observation = np.append(observation, self.data.iloc[self.current_step]['ema20'])
        observation = np.append(observation, self.data.iloc[self.current_step]['ema50'])
        observation = np.append(observation, self.data.iloc[self.current_step]['rsi'])
        observation = np.append(observation, self.data.iloc[self.current_step]['macd'])
        observation = np.append(observation, self.data.iloc[self.current_step]['macd_signal'])
        observation = np.append(observation, self.data.iloc[self.current_step]['macd_hist'])
        observation = np.append(observation, self.data.iloc[self.current_step]['volume'])
        observation = np.append(observation, self.data.iloc[self.current_step]['ADX'])
        observation = np.append(observation, self.data.iloc[self.current_step]['c_DI'])
        observation = np.append(observation, self.data.iloc[self.current_step]['t_DI'])
        observation = np.append(observation, self.data.iloc[self.current_step]['close'])

        # Đảm bảo tất cả các giá trị trong observation đều là số thực
        observation = observation.astype(np.float32)

        # Kiểm tra NaN hoặc vô cùng trong observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print("Warning: NaN or Inf values found in observation")
            # Nếu có NaN hoặc Inf, có thể thay thế bằng giá trị mặc định (ví dụ: 0)
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation


    def reset(self, seed=None):
        """Đặt lại môi trường về trạng thái ban đầu"""
        self.current_step = 0
        self.balance = 100
        self.entry_price = 0
        self.position = 0
        self.pip = 0
        self.win = 0
        self.dem_sell = 0
        self.dem_buy = 0
        self.loss_streak = 0
        self.dem = 0
        self.win_streak = 0
        return self._next_observation(), {}

    def render(self, mode='human'):
        """Hiển thị trạng thái hiện tại của môi trường giao dịch"""
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Win Streak: {self.win_streak}')
        print(f'Loss Streak: {self.loss_streak}')
        print('---')
