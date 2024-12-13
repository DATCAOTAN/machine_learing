import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv
import os
import pandas as pd
import logging
import logging.config
import yaml

# Load cấu hình logging
with open("config/logging.yaml", "r", encoding='utf-8') as file:
    logging_config = yaml.safe_load(file)
    logging.config.dictConfig(logging_config)

# Tạo logger riêng cho phần hyperparameter optimization
hyperparameter_logger = logging.getLogger("hyperparameter_logger")

# Tạo logger riêng cho phần huấn luyện
train_logger = logging.getLogger("training")

def optimize_ppo(trial, data):
    hyperparameter_logger.info("Starting hyperparameter optimization trial.")

    # Định nghĩa các siêu tham số cho PPO
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.95)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)

    hyperparameter_logger.debug(
        f"Trial hyperparameters: n_steps={n_steps}, gamma={gamma}, "
        f"learning_rate={learning_rate}, clip_range={clip_range}, "
        f"gae_lambda={gae_lambda}, ent_coef={ent_coef}, vf_coef={vf_coef}"
    )

    # Tạo môi trường huấn luyện
    env = DummyVecEnv([lambda: TradingEnv(data)])
    hyperparameter_logger.info("Environment initialized.")

    # Tạo mô hình PPO
    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                clip_range=clip_range,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef)
    hyperparameter_logger.info("Model initialized.")

    # Huấn luyện mô hình
    hyperparameter_logger.info("Starting model training...")
    model.learn(total_timesteps=5000)

    # Đánh giá mô hình
    obs = env.reset()
    balance = 100  # Số dư khởi tạo
    hyperparameter_logger.info("Starting model evaluation...")
    for _ in range(len(data) // 2):  # Chạy trên một nửa dữ liệu
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        balance += rewards
        if done:
            obs = env.reset()
    hyperparameter_logger.info(f"Trial completed with balance: {balance}")

    return balance

def train_data():
    model_path = r"models\ppo_trading_xauusd.zip"
    print(model_path)
    train_logger.info("Starting model training process.")

    # Kiểm tra mô hình đã tồn tại chưa
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        train_logger.info(f"Model loaded from {model_path}.zip")
    else:
        train_logger.info("No pre-trained model found. Initializing new model.")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: TradingEnv(pd.DataFrame(), render_mode='human')]), verbose=1)

    # Duyệt qua các tệp dữ liệu từ 2015
    directory = r'data\processed\du_lieu_phan_tich'
    for year in range(2015, 2016):
        for month in range(1, 2):
            if month < 10:
                file_path = os.path.join(directory, str(year),'du_lieu_vang_phan_tich_'+str(year)+'_0'+str(month)+'.csv')
            else:
                file_path = os.path.join(directory,str(year),'du_lieu_vang_phan_tich_'+str(year)+'_'+str(month)+'.csv')

            if os.path.exists(file_path):
                train_logger.info(f"Loading data from {file_path}")
                data = pd.read_csv(file_path)

                if data.isnull().values.any():
                    train_logger.warning(f"Data contains NaN values in {file_path}. Dropping rows with NaN.")
                    data = data.dropna()

                # Tạo môi trường và huấn luyện mô hình
                env = DummyVecEnv([lambda: TradingEnv(data, render_mode='human')])
                timesteps = len(data) // 300
                train_logger.info(f"Training on data with {timesteps} timesteps.")
                model.set_env(env)
                model.learn(total_timesteps=timesteps)
                train_logger.info(f"Training completed on {file_path}")
            else:
                train_logger.warning(f"File not found: {file_path}")

    # Lưu mô hình đã huấn luyện
    model.save(model_path)
    train_logger.info(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_data()
