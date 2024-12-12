

```markdown
# Trading Bot RL PPO

Dự án này triển khai một bot giao dịch sử dụng học tăng cường (Reinforcement Learning) với thuật toán Proximal Policy Optimization (PPO) và thư viện `stable-baselines3`. Bot được thiết kế để giao dịch các tài sản như XAU/USD (Vàng) và BTC/USD (Bitcoin) sử dụng dữ liệu từ MetaTrader5 (MT5).

## Cấu trúc dự án

trading_bot_rl_ppo/
│
├── config/
│   └── logging.yaml            # Cấu hình logging
│
├── data/
│   ├── raw/                    # Dữ liệu thô
│   ├── processed/              # Dữ liệu đã qua xử lý
│   └── README.md               # Thông tin về dữ liệu
│
├── models/                     # Mô hình đã huấn luyện
│   └── ppo_trading_xauusd.zip  # Mô hình PPO đã huấn luyện
│
├── results/                    # Kết quả, logs và lịch sử giao dịch
│   ├── logs/                   # Logs hệ thống
│   │   └── system.log
│   ├── trade_history/          # Lịch sử giao dịch
│   └── evaluation_results.csv  # Kết quả đánh giá mô hình
│
├── scripts/
│   ├── environment.py          # Môi trường giao dịch (TradingEnv)
│   ├── model.py                # Huấn luyện và tối ưu siêu tham số cho PPO
│   ├── live_trading.py         # Giao dịch trực tiếp với mô hình PPO
│   ├── model_inference.py      # Inference với mô hình đã huấn luyện
│   └── README.md               # Hướng dẫn sử dụng cho các script
│
├── notebooks/                  # Jupyter notebooks cho các thí nghiệm
│   ├── 01-train-model.ipynb    # Huấn luyện mô hình
│   ├── 02-optimize-ppo.ipynb   # Tối ưu siêu tham số
│   └── 03-live-trade.ipynb     # Giao dịch trực tiếp với mô hình
│
├── requirements.txt            # Các thư viện Python yêu cầu
├── README.md                   # Tổng quan về dự án và hướng dẫn cài đặt
└── main.py                     # Script để chạy toàn bộ quy trình
```

## Cài đặt

### Bước 1: Clone dự án về máy

```bash
git clone https://github.com/yourusername/trading_bot_rl_ppo.git
cd trading_bot_rl_ppo
```

### Bước 2: Cài đặt môi trường ảo (`venv`)

Trước khi cài đặt các thư viện yêu cầu, bạn cần tạo môi trường ảo để tránh xung đột giữa các thư viện:

1. **Tạo môi trường ảo**:
   - **Windows**:
     ```bash
     python -m venv venv
     ```
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     ```

2. **Kích hoạt môi trường ảo**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

### Bước 3: Cài đặt các thư viện phụ thuộc

Cài đặt các thư viện Python yêu cầu bằng cách chạy:

```bash
pip install -r requirements.txt
```

### Bước 4: Tải MetaTrader5 và thiết lập tài khoản giao dịch

- Cài đặt MetaTrader5 (MT5) trên máy tính của bạn và thiết lập tài khoản giao dịch.
- Đảm bảo rằng phần mềm MT5 đang chạy và kết nối với tài khoản khi bạn chạy bot.

### Bước 5: Cấu hình logging

Đảm bảo rằng file `config/logging.yaml` đã được cấu hình đúng để ghi log. Bạn có thể thay đổi các mức log và định dạng trong file này.

## Cách sử dụng

### Huấn luyện mô hình

Bạn có thể huấn luyện mô hình bằng cách sử dụng script `model.py`. Mô hình sẽ được huấn luyện trên dữ liệu và lưu lại mô hình sau khi hoàn thành.

Để huấn luyện mô hình:

```bash
python scripts/model.py
```

Điều này sẽ bắt đầu quá trình huấn luyện và ghi lại chi tiết về quá trình huấn luyện, bao gồm tối ưu siêu tham số và kết quả huấn luyện. Mô hình huấn luyện sẽ được lưu trong thư mục `models`.

### Tối ưu siêu tham số

Thư viện `optuna` được sử dụng để tối ưu các siêu tham số cho mô hình PPO. Để tối ưu các siêu tham số, chạy lệnh sau:

```bash
python scripts/model.py
```

Điều này sẽ chạy quá trình tối ưu và ghi lại các siêu tham số cho mỗi thử nghiệm.

### Giao dịch trực tiếp

Để thực hiện giao dịch trực tiếp với mô hình đã huấn luyện, sử dụng script `live_trading.py`. Bot sẽ sử dụng mô hình PPO để đưa ra dự đoán và thực hiện giao dịch trực tiếp.

Để bắt đầu giao dịch trực tiếp:

```bash
python scripts/live_trading.py
```

Đảm bảo rằng MetaTrader5 đang chạy và kết nối.

## Logging

Cấu hình logging được lưu trong `config/logging.yaml`. Có các log riêng cho hệ thống, lỗi giao dịch, tối ưu siêu tham số và huấn luyện. Bạn có thể thay đổi mức độ logging và các địa chỉ lưu log trong file này.

### Các loại log:
- **Log live trade**: Lưu trong `results/logs/live_trade.log`.
- **Lỗi giao dịch**: Lưu trong `results/logs/trading_errors.log`.
- **Log tối ưu siêu tham số**: Lưu trong `results/logs/hyperparameter.log`.
- **Log huấn luyện**: Lưu trong `results/logs/training.log`.

## Đánh giá mô hình

Bạn có thể đánh giá hiệu suất của mô hình đã huấn luyện bằng cách sử dụng script `model_inference.py`. Script này cho phép bạn chạy mô hình đã huấn luyện trên dữ liệu chưa thấy để kiểm tra hiệu suất của mô hình.

Để đánh giá mô hình:

```bash
python scripts/model_evaluation.ipynb
```

### Kết quả đánh giá

Kết quả đánh giá sẽ được lưu trong file `results/evaluation_results.csv`.

## Kết luận

Dự án này được thiết kế để hỗ trợ phát triển các bot giao dịch sử dụng học tăng cường. Nó sử dụng PPO để tối ưu chính sách và tích hợp với MetaTrader5 cho việc giao dịch trực tiếp. Bạn có thể mở rộng bot bằng cách huấn luyện lại trên dữ liệu mới, tối ưu các siêu tham số hoặc thử nghiệm với các chiến lược giao dịch khác nhau.

## Lưu ý thêm

1. **Chuẩn bị dữ liệu**: Đảm bảo rằng dữ liệu đã được xử lý và làm sạch trước khi huấn luyện mô hình. Dữ liệu phải có định dạng CSV với các chỉ báo kỹ thuật như RSI, MACD, EMA và ADX.
   
2. **Tuning mô hình**: Bạn có thể tinh chỉnh mô hình PPO bằng cách thay đổi các siêu tham số trong `model.py`. Optuna được sử dụng để tìm kiếm bộ siêu tham số tốt nhất.

3. **Giao dịch trực tiếp**: Để giao dịch trực tiếp, đảm bảo rằng terminal MT5 của bạn đã được kết nối và được cấu hình để thực hiện các giao dịch dựa trên dự đoán của mô hình PPO.

4. **Các phụ thuộc**: Dự án sử dụng `stable-baselines3` cho thuật toán PPO, `ta-lib` cho các chỉ báo kỹ thuật và `MetaTrader5` cho tích hợp giao dịch.



### Các tính năng chính:
- **Tối ưu siêu tham số**: Tối ưu siêu tham số của mô hình PPO bằng Optuna.
- **Logging**: Cấu hình ghi log chi tiết cho huấn luyện, tối ưu siêu tham số và giao dịch trực tiếp.
- **Giao dịch trực tiếp**: Tích hợp với MetaTrader5 để thực hiện giao dịch trực tiếp dựa trên dự đoán của mô hình.
- **Đánh giá mô hình**: Cho phép đánh giá hiệu suất của mô hình trên dữ liệu chưa thấy.

