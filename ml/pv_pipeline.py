import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from app import config

class PVModelPipeline:
    """
    一個整合了PVLib的數位孿生模型管道，負責從資料載入、模型建立到預測的整個流程。

    這個類別在初始化時會根據設定檔建立一個完整的 ModelChain，
    並提供一個 run 方法來執行預測。
    """
    def __init__(self, ):
        """初始化 PVModelPipeline，建立 ModelChain 模型"""
        print("正在初始化 PVModelPipeline...")

        # 1. 建立 Location 物件
        self.location = Location(
            latitude=config.SITE_LATITUDE, 
            longitude=config.SITE_LONGITUDE, 
            altitude=config.SITE_ALTITUDE, 
            name=config.SITE_NAME
        )
        print(f"場域建立成功: {self.location.name}")

        # 2. 載入逆變器資料庫並選擇指定的逆變器
        try:
            inverter_database = pvlib.pvsystem.retrieve_sam(name=config.INVERTER_DATABASE)
            self.inverter_parameter = inverter_database[config.INVERTER_NAME]
            print(f"選擇的逆變器: {config.INVERTER_NAME}")
        except Exception as e:
            print(f"載入逆變器資料庫失敗: {e}")
            raise

        # 3. 建立太陽能板系統物件
        self.system = PVSystem(
            surface_tilt=config.SURFACE_TILT,
            surface_azimuth=config.SURFACE_AZIMUTH,
            module_parameters=config.MODULE_PARAMETERS,
            inverter_parameters=self.inverter_parameter,
            temperature_model_parameters=config.TEMPERATURE_PARAMETERS,
            modules_per_string=config.MODULES_PER_STRING,
            strings_per_inverter=config.STRINGS_PER_INVERTER
        )
        print("PVSystem 建立成功")

        # 4. 建立 ModelChain 模型
        self.modelchain = ModelChain(
            self.system, 
            self.location,
            aoi_model='no_loss',
            spectral_model='no_loss',
        )
        print("ModelChain 建立成功")

    def run(self, weather_df):
        """
        使用已經初始化好的 ModelChain 執行發電量預測。

        Args:
        weather_data (pd.DataFrame): 包含必要欄位的天氣資料 DataFrame，索引為 datetimeIndex，且包含以下欄位：
            - temp_air
            - temp_dew
            - ghi
            - dni
            - dhi
            - wind_speed
            - wind_direction
            - albedo
            - pressure

        Returns:
            pd.DataFrame: 包含預測結果的 DataFrame，索引為 datetimeIndex，且包含以下欄位：
                - ac: 預測的交流電能量 (kWh)
        """

        print("正在執行 PVModelPipeline 的 run 方法...")
        
        # 檢查傳入的 weather_df 是否為空
        if weather_df.empty:
            print("警告: 傳入的天氣資料 DataFrame 為空，無法進行預測。")
            return pd.DataFrame()  # 返回一個空的 DataFrame
        
        # 執行預測
        self.modelchain.run_model(weather_df)
        print("預測完成，正在處理結果...")
        
        return self.modelchain.results.ac
    

# --- 測試程式碼 ---
if __name__ == "__main__":
    # 測試 PVModelPipeline 的功能
    pipeline = PVModelPipeline()

    # 載入測試用的天氣資料
    weather_data_path = "C:/Users/Pei/OneDrive/桌面/GitHub/pv_forecast/data/processed/processed_weather_data.csv"
    weather_df = pd.read_csv(weather_data_path, index_col='datetime', parse_dates=True)
    print("測試用的天氣資料載入成功，前5行:")
    print(weather_df.head())

    # 執行預測
    predicted_ac_energy = pipeline.run(weather_df)
    print("預測的交流電能量 (前5行):")
    print(predicted_ac_energy.head())