import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv
import os
import pandas as pd

def optimize_ppo(trial, data):
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.95)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)
    
    # Thiết lập môi trường
    env = DummyVecEnv([lambda: TradingEnv(data)])
    
    # Tạo mô hình PPO với các siêu tham số được chọn
    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                clip_range=clip_range,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef)
    
    # Huấn luyện mô hình trong một khoảng thời gian ngắn để đánh giá siêu tham số
    model.learn(total_timesteps=5000)
    
    # Đánh giá mô hình dựa trên lợi nhuận cuối cùng
    obs = env.reset()
    balance = 100  # Khởi tạo số dư ban đầu
    for _ in range(len(data) // 2):  # Kiểm tra trên một nửa dữ liệu
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        balance += rewards
        if done:
            obs = env.reset()
    return balance

def train_data():
    model_path = "models/ppo_trading_xauusd"
    
    # Kiểm tra xem mô hình đã huấn luyện trước đó có tồn tại không
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print(f"Đã tải mô hình từ {model_path}.zip")
    else:
        # Nếu mô hình chưa tồn tại, khởi tạo mô hình mới
        print("Mô hình chưa tồn tại, khởi tạo mô hình mới.")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: TradingEnv(pd.DataFrame(), render_mode='human')]), verbose=1)

    # Duyệt qua các file dữ liệu từ tháng 1 đến tháng 12 của các năm
    for year in range(2015, 2025):
        for month in range(1, 13):
            # Cập nhật đường dẫn file dữ liệu theo thư mục processed
            if month < 10:
                file_path = f"C:\\Users\\nguye\\OneDrive\\documents\\python\\trading_bot_rl_ppo\\data\\processed\\du_lieu_phan_tich\\{year}\\du_lieu_vang_phan_tich_{year}_0{month}.csv"
            else:
                file_path = f"C:\\Users\\nguye\\OneDrive\\documents\\python\\trading_bot_rl_ppo\\data\\processed\\du_lieu_phan_tich\\{year}\\du_lieu_vang_phan_tich_{year}_{month}.csv"

            # Kiểm tra sự tồn tại của file
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)

                # Kiểm tra và xử lý các giá trị NaN trong dữ liệu
                if data.isnull().values.any():
                    print(f"Warning: Dữ liệu chứa giá trị NaN trong {file_path}. Đang loại bỏ các dòng NaN.")
                    data = data.dropna()  # Loại bỏ các dòng chứa NaN

                # Tạo môi trường với dữ liệu hiện tại
                env = DummyVecEnv([lambda: TradingEnv(data, render_mode='human')])

                # Tiếp tục huấn luyện mô hình trên dữ liệu hiện tại
                timesteps = len(data) // 300  # Tính số bước huấn luyện dựa trên kích thước dữ liệu
                model.set_env(env)
                model.learn(total_timesteps=timesteps)  # Huấn luyện mô hình

                print(f"Đã huấn luyện xong trên dữ liệu: {file_path}")
            else:
                print(f"File không tồn tại: {file_path}")

    # Lưu mô hình đã huấn luyện lại sau khi hoàn thành vòng lặp qua các file dữ liệu
    model.save("ppo_trading_xauusd")
    print("Mô hình đã được lưu lại.")
