import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from app import config
from ml.model_trainer import MLBiasCorrector

class PVModelPipeline:
    """
    一個整合了PVLib的數位孿生模型管道，負責從資料載入、模型建立到預測的整個流程。

    這個類別在初始化時會根據設定檔建立一個完整的 ModelChain，
    並提供一個 run 方法來執行預測。
    """
    def __init__(self, use_ml_correction=True, model_path='models/bias_corrector.pkl'):
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

        # 5. 初始化 ML 偏差修正器（可選）
        self.use_ml_correction = use_ml_correction
        self.bias_corrector = None
        if self.use_ml_correction:
            self.bias_corrector = MLBiasCorrector(model_path=model_path)
            try:
                self.bias_corrector.load_model()
                print("ML 偏差修正模型載入成功")
            except FileNotFoundError:
                print("警告: 尚未找到 ML 偏差修正模型，將只回傳 PVLib 預測")

    def _predict_physics(self, weather_df):
        """執行 PVLib 物理模型預測。"""
        self.modelchain.run_model(weather_df)
        return self.modelchain.results.ac.rename('pvlib_ac')

    def _apply_bias_correction(self, weather_df, pvlib_ac):
        """套用 ML 偏差修正，回傳修正後預測。"""
        if self.bias_corrector is None or not self.bias_corrector.is_trained:
            return None

        feature_df = self.bias_corrector.build_prediction_features(weather_df, pvlib_ac)
        correction = pd.Series(
            self.bias_corrector.predict_correction(feature_df),
            index=pvlib_ac.index,
            name='ml_correction'
        )
        corrected = (pvlib_ac + correction).rename('corrected_ac')

        return pd.concat([pvlib_ac, correction, corrected], axis=1)

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
            pd.DataFrame: 包含預測結果，至少有 `pvlib_ac`。
            若成功載入 ML 模型，則另包含 `ml_correction` 與 `corrected_ac`。
        """

        print("正在執行 PVModelPipeline 的 run 方法...")
        
        # 檢查傳入的 weather_df 是否為空
        if weather_df.empty:
            print("警告: 傳入的天氣資料 DataFrame 為空，無法進行預測。")
            return pd.DataFrame()  # 返回一個空的 DataFrame
        
        # 先做物理模型預測
        pvlib_ac = self._predict_physics(weather_df)
        print("預測完成，正在處理結果...")

        # 再做機器學習偏差修正
        if self.use_ml_correction:
            corrected_result = self._apply_bias_correction(weather_df, pvlib_ac)
            if corrected_result is not None:
                return corrected_result
        
        return pd.DataFrame({'pvlib_ac': pvlib_ac})
    

# --- 測試程式碼 ---
if __name__ == "__main__":
    # 測試 PVModelPipeline 的功能
    from ml.data_loader import load_custom_weather_data
    pipeline = PVModelPipeline()

    # 載入測試用的天氣資料
    weather_data_path = "C:/Users/Pei/OneDrive/桌面/GitHub/pv_forecast/data/processed/processed_weather_data.csv"
    weather_df = load_custom_weather_data(weather_data_path)
    print("測試用的天氣資料載入成功，前5行:")
    print(weather_df.head())

    # 執行預測
    predicted_ac_energy = pipeline.run(weather_df)
    print("預測的交流電能量 (前5行):")
    print(predicted_ac_energy.head())