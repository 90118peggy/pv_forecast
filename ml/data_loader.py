import pandas as pd
import numpy as np

def load_custom_weather_data(file_path, timezone='Etc/GMT+5'):
    """
    從 CSV 文件中載入天氣資料，並進行必要的預處理。
    
    參數:
    file_path (str): CSV 文件的路徑
    timezone (str): 時區信息，預設為 'Etc/GMT+5' (美東標準時間)
    
    返回:
    pd.DataFrame: 處理後的天氣資料
    """
    # 讀取 CSV 文件
    weather_data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    
    # 確保所有必要的欄位都存在
    required_columns = ['temp_air', 
                        'temp_dew', 
                        'ghi', 'dni', 'dhi', 
                        'wind_speed', 'wind_direction', 
                        'albedo', 'pressure']
    for col in required_columns:
        if col not in weather_data.columns:
            raise ValueError(f"缺少必要的欄位: {col}")
    
    # 填補缺失值（如果有）
    weather_data.ffill(inplace=True)  # 向前填補
    weather_data.bfill(inplace=True)  # 向後填補

    # --- 時區處理 (如果需要) ---
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(timezone)
    
    return weather_data

def load_nsrdb_weather_data(email, api_key, latitude, longitude, year, timezone='Etc/GMT+5'):
    """
    從 NSRDB 格式的 CSV 文件中載入天氣資料，並進行必要的預處理。
    
    參數:
    file_path (str): NSRDB CSV 文件的路徑
    
    返回:
    pd.DataFrame: 處理後的天氣資料
    """
    from pvlib import iotools

    weather_data, metadata = iotools.get_nsrdb_psm4_tmy(
                                                        latitude=latitude,
                                                        longitude=longitude,
                                                        api_key=api_key, 
                                                        email=email, 
                                                        year=year,
                                                        )
    
    # 確保所有必要的欄位都存在
    required_columns = ['temp_air', 
                        'temp_dew', 
                        'ghi', 'dni', 'dhi', 
                        'wind_speed', 'wind_direction', 
                        'albedo', 'pressure']
    for col in required_columns:
        if col not in weather_data.columns:
            raise ValueError(f"缺少必要的欄位: {col}")
    
    # 填補缺失值（如果有）
    weather_data.fillna(method='ffill', inplace=True)  # 向前填補
    weather_data.fillna(method='bfill', inplace=True)  # 向後填補
    
    # --- 時區處理 (如果需要) ---
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(timezone)

    return weather_data