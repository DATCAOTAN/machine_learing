import os
import pandas as pd
import shutil
import ta  # Thư viện technical analysis
import numpy as np

def add_column_names_and_save():
    """Thêm tên cột và xử lý dữ liệu."""
    for year in range(2015, 2016):
        for month in range(1, 2):
            # Xác định đường dẫn file từ thư mục mới
            if month < 10:
                absolute_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\raw\XAUUSD(2015-2024)\{year}\XAUUSD_{year}_0{month}.csv"
            else:
                absolute_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\raw\XAUUSD(2015-2024)\{year}\XAUUSD_{year}_{month}.csv"
            
            # Kiểm tra sự tồn tại của file để tránh lỗi
            if os.path.exists(absolute_path):
                try:
                    # Đọc file CSV với header và tên cột mới
                    data = pd.read_csv(absolute_path, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Tạo cột datetime và đặt làm chỉ mục
                    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y.%m.%d %H:%M')
                    data.set_index('datetime', inplace=True)
                    
                    # Chọn các cột cần thiết
                    data = data[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Đặt tên file tạm để ghi dữ liệu tạm thời
                    temp_name = f"temp_XAUUSD_{year}_{month}.csv"
                    data.to_csv(temp_name, index=True)
                    
                    # Di chuyển file tạm để ghi đè lên file gốc
                    shutil.move(temp_name, absolute_path)
                    print(f"Đã xử lý và lưu file: {absolute_path}")
                
                except Exception as e:
                    print(f"Lỗi khi xử lý file {absolute_path}: {e}")
            
            else:
                print(f"File không tồn tại: {absolute_path}")

def calculate_dmi(data, period=14):
    """Tính DMI (Directional Movement Index)"""
    # Tính True Range (TR)
    data['high_diff'] = data['high'].diff()
    data['low_diff'] = data['low'].diff()
    data['tr'] = data[['high_diff', 'low_diff', 'close']].max(axis=1).fillna(0)

    # Tính +DI và -DI
    data['c_DM'] = np.where((data['high_diff'] > data['low_diff']) & (data['high_diff'] > 0), data['high_diff'], 0)
    data['t_DM'] = np.where((data['low_diff'] > data['high_diff']) & (data['low_diff'] > 0), data['low_diff'], 0)

    # Tính toán các chỉ số cần thiết
    data['TR14'] = data['tr'].rolling(window=period).mean()
    data['c_DM14'] = data['c_DM'].rolling(window=period).mean()
    data['t_DM14'] = data['t_DM'].rolling(window=period).mean()

    data['c_DI'] = (data['c_DM14'] / data['TR14']) * 100
    data['t_DI'] = (data['t_DM14'] / data['TR14']) * 100

    return data

def combined_trend(data):
    """Tính các xu hướng dựa trên ADX và các chỉ báo kỹ thuật khác"""
    # Tính ADX
    adx_period = 14
    data['ADX'] = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=adx_period).adx()

    # Tính DMI
    data = calculate_dmi(data, period=adx_period)

    # Khởi tạo cột Trend
    data['Trend'] = 'No Trend'

    # Đánh dấu xu hướng tăng
    data.loc[
        (data['ADX'] > 25) & 
        (data['c_DI'] > data['t_DI']) & 
        (data['macd'] > data['macd_signal']) & 
        (data['rsi'] < 70),
        'Trend'
    ] = 'Uptrend'

    # Đánh dấu xu hướng giảm
    data.loc[
        (data['ADX'] > 25) & 
        (data['t_DI'] > data['c_DI']) & 
        (data['macd'] < data['macd_signal']) & 
        (data['rsi'] > 30),
        'Trend'
    ] = 'Downtrend'

    # Xu hướng không rõ
    data.loc[data['ADX'] < 25, 'Trend'] = 'Sideways'

    return data

def preprocess_data():
    """Tiền xử lý và lưu trữ các dữ liệu đã được xử lý với các chỉ báo"""
    for year in range(2015, 2026):
        for month in range(1, 2 ):
            # Đường dẫn file và file đầu ra
            if month < 10:
                absolute_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\raw\XAUUSD(2015-2024)\{year}\XAUUSD_{year}_0{month}.csv"        
                output_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\processed\du_lieu_phan_tich_{year}_0{month}.csv"
            else:
                absolute_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\raw\XAUUSD(2015-2024)\{year}\XAUUSD_{year}_{month}.csv"
                output_path = rf"C:\Users\nguye\OneDrive\documents\python\trading_bot_rl_ppo\data\processed\du_lieu_phan_tich_{year}_{month}.csv"
            
            # Kiểm tra sự tồn tại của file
            if os.path.exists(absolute_path):
                if os.path.exists(absolute_path):
                    try:
                        data = pd.read_csv(absolute_path)
                        data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()

                        # Tính toán MACD (Moving Average Convergence Divergence)
                        macd = ta.trend.MACD(close=data['close'])
                        data['macd'] = macd.macd()
                        data['macd_signal'] = macd.macd_signal()
                        data['macd_hist'] = macd.macd_diff()

                        # Tính toán EMA (Exponential Moving Average)
                        ema10_window = 10
                        ema20_window = 20
                        ema50_window = 50

                        data['ema10'] = ta.trend.EMAIndicator(close=data['close'], window=ema10_window).ema_indicator()
                        data['ema20'] = ta.trend.EMAIndicator(close=data['close'], window=ema20_window).ema_indicator()
                        data['ema50'] = ta.trend.EMAIndicator(close=data['close'], window=ema50_window).ema_indicator()

                        # Tính toán Fibonacci retracement levels
                        rolling_window = 50
                        data['highest_high'] = data['high'].rolling(window=rolling_window).max()
                        data['lowest_low'] = data['low'].rolling(window=rolling_window).min()

                        # Fibonacci levels (23.6%, 38.2%, 50%, 61.8%)
                        data['fibonacci_23.6'] = data['highest_high'] - (data['highest_high'] - data['lowest_low']) * 0.236
                        data['fibonacci_38.2'] = data['highest_high'] - (data['highest_high'] - data['lowest_low']) * 0.382
                        data['fibonacci_50.0'] = data['highest_high'] - (data['highest_high'] - data['lowest_low']) * 0.5
                        data['fibonacci_61.8'] = data['highest_high'] - (data['highest_high'] - data['lowest_low']) * 0.618 

                        def calculate_dmi(data, period=14):
                            # Tính True Range (TR)
                            data['high_diff'] = data['high'].diff()
                            data['low_diff'] = data['low'].diff()
                            data['tr'] = data[['high_diff', 'low_diff', 'close']].max(axis=1).fillna(0)
                            
                            # Tính +DI và -DI
                            data['c_DM'] = np.where((data['high_diff'] > data['low_diff']) & (data['high_diff'] > 0), data['high_diff'], 0)
                            data['t_DM'] = np.where((data['low_diff'] > data['high_diff']) & (data['low_diff'] > 0), data['low_diff'], 0)
                            
                            # Tính toán các chỉ số cần thiết
                            data['TR14'] = data['tr'].rolling(window=period).mean()
                            data['c_DM14'] = data['c_DM'].rolling(window=period).mean()
                            data['t_DM14'] = data['t_DM'].rolling(window=period).mean()
                            
                            data['c_DI'] = (data['c_DM14'] / data['TR14']) * 100
                            data['t_DI'] = (data['t_DM14'] / data['TR14']) * 100

                            return data

                        # Sử dụng hàm calculate_dmi trong combined_trend
                        def combined_trend(data):
                            # Tính ADX
                            adx_period = 14  
                            data['ADX'] = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=adx_period).adx()

                            # Tính DMI
                            data = calculate_dmi(data, period=adx_period)

                            # Khởi tạo cột Trend
                            data['Trend'] = 'No Trend'
                            
                            # Đánh dấu xu hướng tăng
                            data.loc[
                                (data['ADX'] > 25) & 
                                (data['c_DI'] > data['t_DI']) & 
                                (data['macd'] > data['macd_signal']) & 
                                (data['rsi'] < 70),
                                'Trend'
                            ] = 'Uptrend'

                            # Đánh dấu xu hướng giảm
                            data.loc[
                                (data['ADX'] > 25) & 
                                (data['t_DI'] > data['c_DI']) & 
                                (data['macd'] < data['macd_signal']) & 
                                (data['rsi'] > 30),
                                'Trend'
                            ] = 'Downtrend'

                            # Xu hướng không rõ
                            data.loc[data['ADX'] < 25, 'Trend'] = 'Sideways'

                            return data


                        # Tính toán mức độ biến động lịch sử
                        def calculate_historical_volatility(data, window=20):
                            # Tính log returns
                            log_returns = np.log(data['close'] / data['close'].shift(1))
                            # Tính độ lệch chuẩn của log returns trong khoảng thời gian xác định
                            hv = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized HV
                            return hv

                        # Xác định mức độ biến động cho mỗi hàng dữ liệu
                        def determine_volatility_level(data, window=20):
                            hv = calculate_historical_volatility(data, window)
                            mean_hv = hv.mean()  # Giá trị trung bình của HV
                            std_hv = hv.std()    # Độ lệch chuẩn của HV
                            
                            # Tạo một cột mới để lưu mức độ biến động
                            volatility_levels = []

                            for current_hv in hv:
                                if current_hv > mean_hv + std_hv:
                                    volatility_levels.append("High Volatility")
                                elif current_hv < mean_hv - std_hv:
                                    volatility_levels.append("Low Volatility")
                                else:
                                    volatility_levels.append("Moderate Volatility")
                            
                            return volatility_levels

                        # Áp dụng hàm xác định mức độ biến động cho toàn bộ DataFrame
                        data['HV'] = determine_volatility_level(data)
                        data = combined_trend(data)
                        data.dropna(inplace=True)
                        
                        data.to_csv(output_path)

                        print(f"Dữ liệu đã được lưu vào '{output_path}' với các chỉ báo và phân tích.")
                    
                    except Exception as e:
                        print(f"Lỗi khi xử lý file {absolute_path}: {e}")
        
            else:
                print(f"File không tồn tại: {absolute_path}")


