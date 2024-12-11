from setuptools import setup, find_packages

setup(
    name="trading_bot_rl_ppo",  # Tên dự án
    version="1.0.0",            # Phiên bản
    author="Your Name",         # Tên tác giả
    author_email="your_email@example.com",  # Email liên hệ
    description="A trading bot using reinforcement learning with PPO.",  # Mô tả ngắn về dự án
    long_description=open("README.md").read(),  # Đọc mô tả chi tiết từ README.md
    long_description_content_type="text/markdown",  # Định dạng của phần mô tả chi tiết
    url="https://github.com/yourusername/trading_bot_rl_ppo",  # URL dự án (nếu có)
    packages=find_packages(),  # Tự động tìm các thư mục chứa mã nguồn Python
    install_requires=[         # Danh sách thư viện cần thiết
        "gym==0.26.2",
        "stable-baselines3==1.8.0",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "ta",
        "MetaTrader5",
        "optuna",  # Thêm optuna cho tối ưu hóa siêu tham số
    ],
    classifiers=[              # Phân loại dự án
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",    # Yêu cầu phiên bản Python >= 3.8
    entry_points={             # Điểm vào lệnh nếu muốn chạy trực tiếp
        "console_scripts": [
            "trade-bot=live_trading:main",  # Lệnh 'trade-bot' sẽ chạy hàm main() trong live_trading.py
        ],
    },
)
