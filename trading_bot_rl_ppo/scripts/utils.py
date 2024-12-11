import os
from stable_baselines3 import PPO

def check_file_exists(file_path):
    """Kiểm tra xem file có tồn tại không."""
    if not os.path.exists(file_path):
        print(f"File không tồn tại: {file_path}")
        return False
    return True

def save_data_to_csv(data, file_path):
    """Lưu dữ liệu vào file CSV."""
    data.to_csv(file_path, index=False)
    print(f"Đã lưu dữ liệu vào {file_path}")

def save_model(model, filename):
    """Lưu mô hình vào file"""
    model.save(filename)
    print(f"Đã lưu mô hình vào {filename}")

def load_model(filename):
    """Tải mô hình từ file"""
    return PPO.load(filename)
