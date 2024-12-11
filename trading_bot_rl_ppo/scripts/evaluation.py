import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv

def evaluate_model(model, data):
    """Đánh giá mô hình và tính toán các chỉ số hiệu suất"""
    env = DummyVecEnv([lambda: TradingEnv(data)])
    rewards = []
    for _ in range(len(data) - 1):
        action, _ = model.predict(env.reset())
        rewards.append(action)
    total_return = np.sum(rewards)
    sharpe_ratio = np.mean(rewards) / np.std(rewards)  # Tính Sharpe ratio
    print(f'Total Return: {total_return}')
    print(f'Sharpe Ratio: {sharpe_ratio}')
