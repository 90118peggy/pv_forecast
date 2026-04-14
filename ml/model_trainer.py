import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os


class MLBiasCorrector:
    """
    機器學習偏差修正模型
    
    用途：學習 PVLib 物理預測和實際發電量之間的偏差，並進行修正。
    """
    
    def __init__(self, model_path='models/bias_corrector.pkl'):
        """
        初始化偏差修正模型

        Args:
            model_path: 模型儲存路徑，預設為 'models/bias_corrector.pkl'
        """

        self.model_path = model_path
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_trained = False
        self.daytime_config = {
            'pvlib_threshold': 0.00,  # PVLib 預測值大於此閾值視為白天
            'ghi_threshold': 0.1,    # GHI 大於此閾值視為白天
            'use_ghi': True,         # 是否使用 GHI 作為輔助判斷
            'use_time_window': False,  # 是否啟用時段判斷
            'day_start_hour': 5,     # 白天開始小時 (含)
            'day_end_hour': 19       # 白天結束小時 (不含)
        }

    def prepare_training_data(self, weather_df, pvlib_predictions, actual_power):
        """
        準備訓練數據：結合天氣數據、物理預測和實際功率
        
        Args:
            weather_df (pd.DataFrame): 天氣資料
            pvlib_predictions (pd.Series): PVLib 物理預測值
            actual_power (pd.Series): 實際發電功率
            
        Returns:
            tuple: (X_features, y_residuals)
        """
        # 合併資料
        training_data = pd.concat([
            weather_df,
            pvlib_predictions.rename('pvlib_prediction'),
            actual_power.rename('actual_power')
        ], axis=1)
        
        # 移除含有 NaN 的列
        training_data = training_data.dropna()
        
        # 計算殘差（偏差）
        training_data['residual'] = training_data['actual_power'] - training_data['pvlib_prediction']
        
        # 選擇特徵
        feature_cols = [col for col in weather_df.columns if col not in ['datetime']]
        feature_cols.append('pvlib_prediction')  # 加入物理預測作為特徵
        
        self.feature_names = feature_cols
        
        X = training_data[feature_cols]
        y = training_data['residual']
        
        return X, y

    def build_prediction_features(self, weather_df, pvlib_predictions):
        """
        依照訓練時的欄位順序建立推論特徵。

        Args:
            weather_df (pd.DataFrame): 天氣資料
            pvlib_predictions (pd.Series): PVLib 物理預測值

        Returns:
            pd.DataFrame: 可直接餵給模型的特徵矩陣
        """
        if not self.feature_names:
            raise ValueError("feature_names 不存在，請先訓練模型或載入模型")

        prediction_data = weather_df.copy()
        prediction_data['pvlib_prediction'] = pvlib_predictions

        missing_cols = [col for col in self.feature_names if col not in prediction_data.columns]
        if missing_cols:
            raise ValueError(f"推論特徵缺少必要欄位: {missing_cols}")

        return prediction_data[self.feature_names]
    
    def build_daytime_mask(self, X):
        return self._build_daytime_mask(X)

    def _split_data(self, X, y, test_size=0.2, split_method='random'):
        """依照指定策略切分訓練/測試資料。"""
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size 必須介於 0.0 到 1.0 之間，目前值：{test_size}")

        if split_method == 'random':
            return train_test_split(X, y, test_size=test_size, random_state=42)

        if split_method == 'time':
            # 時序切分：前段做訓練，後段做測試，避免未來資訊洩漏。
            if len(X) < 2:
                raise ValueError("資料筆數不足，無法做 time split")

            if isinstance(X.index, pd.DatetimeIndex):
                X = X.sort_index()
                y = y.loc[X.index]

            split_idx = int(len(X) * (1 - test_size))
            if split_idx <= 0 or split_idx >= len(X):
                raise ValueError("切分後訓練集或測試集為空，請調整 test_size")

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]
            return X_train, X_test, y_train, y_test

        raise ValueError("split_method 僅支援 'random' 或 'time'")
    
    def _build_daytime_mask(self, 
                            X,
                            ):
        """建立白天資料的遮罩，確保訓練資料只包含白天的數據。
        # 這裡可以根據具體需求實現白天資料的過濾邏輯，例如：
        - 根據 PVLib 預測值是否大於某個閾值來判斷是否為白天
        - 或者結合 GHI 值來進行更精確的判斷
        """
        if 'pvlib_prediction' not in X.columns:
            raise ValueError("X 中缺少 'pvlib_prediction' 欄位，無法根據 PVLib 預測值建立白天遮罩")

        pvlib_mask = X['pvlib_prediction'] >= self.daytime_config['pvlib_threshold']

        masks = [pvlib_mask]

        if self.daytime_config['use_ghi'] and 'ghi' in X.columns:
            ghi_mask = X['ghi'] >= self.daytime_config['ghi_threshold']
            masks.append(ghi_mask)

        if self.daytime_config.get('use_time_window', False):
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("啟用 use_time_window 時，X.index 必須是 DatetimeIndex")

            start_hour = int(self.daytime_config.get('day_start_hour', 5))
            end_hour = int(self.daytime_config.get('day_end_hour', 19))
            hour_series = pd.Series(X.index.hour, index=X.index)

            # 支援跨日區間，例如 20~6。
            if start_hour < end_hour:
                time_mask = (hour_series >= start_hour) & (hour_series < end_hour)
            else:
                time_mask = (hour_series >= start_hour) | (hour_series < end_hour)

            masks.append(time_mask)

        mask = masks[0]
        for extra_mask in masks[1:]:
            mask = mask | extra_mask

        return mask.fillna(False)

        

    def train(self, X, y, test_size=0.2, split_method='random', daytime_only=False):
        """
        訓練偏差修正模型
        
        Args:
            X (pd.DataFrame): 特徵矩陣
            y (pd.Series): 目標變數（殘差）
            test_size: 測試集比例
            split_method: 資料切分方式（'random' 或 'time'）
        """
        print(f"正在訓練偏差修正模型... (split={split_method})")
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = self._split_data(
            X, y, test_size=test_size, split_method=split_method
        )
        
        # 可選：只使用白天資料訓練
        # 使用mask可以讓index對齊，確保X_train和y_train的index一致，避免後續訓練時出現index不匹配的問題。
        if daytime_only:
            mask = self._build_daytime_mask(X_train)
            X_train = X_train.loc[mask]
            y_train = y_train.loc[mask]

        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 評估模型
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"訓練集 MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        print(f"測試集 MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        self.is_trained = True
        
        return {
            'split_method': split_method,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    
    def predict_correction(self, X):
        """
        預測偏差修正值
        
        Args:
            X (pd.DataFrame): 特徵矩陣
            
        Returns:
            np.ndarray: 預測的殘差值
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先執行 train() 方法")
        
        return self.model.predict(X)
    
    def save_model(self):
        """儲存訓練好的模型"""
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'daytime_config': self.daytime_config
        }
        joblib.dump(payload, self.model_path)
        print(f"模型已儲存至: {self.model_path}")
    
    def load_model(self):
        """載入已訓練的模型"""
        if os.path.exists(self.model_path):
            payload = joblib.load(self.model_path)

            # 向後相容：如果舊檔案只儲存 model 物件，仍可正常載入
            if isinstance(payload, dict) and 'model' in payload:
                self.model = payload['model']
                self.daytime_config = payload.get('daytime_config', self.daytime_config)
                self.feature_names = payload.get('feature_names')
            else:
                self.model = payload

            self.is_trained = True
            print(f"模型已從 {self.model_path} 載入")
        else:
            raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")