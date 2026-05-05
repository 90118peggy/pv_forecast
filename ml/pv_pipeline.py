import pandas as pd
import sys
import os
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import config
from ml.model_trainer import MLBiasCorrector

class PVModelPipeline:
    """
    一個整合了PVLib的數位孿生模型管道，負責從資料載入、模型建立到預測的整個流程。

    這個類別在初始化時會根據設定檔建立一個完整的 ModelChain，
    會預先設定是否此次訓練需要使用ML的偏差修正，
    可以設定是否有加入偏差修正，
    雖然是同一個Pipeline但是存在兩種不同的模式，
    分別是純物理模型預測和物理模型加上ML偏差修正的預測，
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
        pvlib_result = self.modelchain.results.ac.rename('pvlib_ac')
        # 調試：看原始值
        print(f"[DEBUG] PVLib 原始預測 (例子片段，單位 W):\n{pvlib_result.head(10)}")
        print(f"[DEBUG] PVLib 預測統計: min={pvlib_result.min()}, max={pvlib_result.max()}, mean={pvlib_result.mean()}")

        # 對於非常小的負值（例如數值上的雜訊或 inverter model 的 sentinel 值），將其裁切為 0
        try:
            pvlib_result = pvlib_result.astype(float)
            neg_count = (pvlib_result < 0).sum()
            if neg_count > 0:
                print(f"[WARN] 偵測到 {neg_count} 個負的 pvlib_ac 值，將裁切為 0。原始最小值: {pvlib_result.min()}")
            pvlib_result = pvlib_result.clip(lower=0.0)
        except Exception as e:
            print(f"[WARN] 無法轉換 pvlib_result 型別或裁切：{e}")

        return pvlib_result

    def _apply_bias_correction(self, weather_df, pvlib_ac):
        """套用 ML 偏差修正，回傳修正後預測。"""
        if self.bias_corrector is None or not self.bias_corrector.is_trained:
            return None

        # 1. 建立完整推論特徵 DataFrame，包含天氣資料和 PVLib 預測值
        feature_df = self.bias_corrector.build_prediction_features(weather_df, pvlib_ac)

        # 2. 建立白天遮罩
        daytime_mask = self.bias_corrector.build_daytime_mask(feature_df)

        # 3. 先建立完整時間軸上的correction，夜間預設為 0
        correction = pd.Series(0.0, index=pvlib_ac.index, name='ml_correction')

        # 4. 只對白天資料進行ML偏差修正預測
        if daytime_mask.any():
            daytime_features = feature_df.loc[daytime_mask]
            daytime_correction = self.bias_corrector.predict_correction(daytime_features)
            correction.loc[daytime_mask] = daytime_correction

        # 5. 計算修正後的預測值
        corrected = (pvlib_ac + correction).rename('corrected_ac')

        return pd.concat([pvlib_ac, 
                          daytime_mask.rename('is_daytime'), 
                          correction, 
                          corrected], axis=1)

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
        # pvlib_ac 的單位為 W，轉換為 kW 以與實際發電數據一致
        pvlib_ac = pvlib_ac / 1000.0
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
    pipeline = PVModelPipeline(use_ml_correction=True, model_path='models/bias_corrector.pkl')

    # 載入測試用的天氣資料
    weather_data_path = "C:/Users/Pei/OneDrive/桌面/GitHub/pv_forecast/data/processed/processed_weather_data.csv"
    weather_df = load_custom_weather_data(weather_data_path)
    print("測試用的天氣資料載入成功，前5行:")
    print(weather_df.head())

    # 執行預測
    predicted_ac_energy = pipeline.run(weather_df)
    print("預測的交流電能量 (前5行):")
    print(predicted_ac_energy[40:50])