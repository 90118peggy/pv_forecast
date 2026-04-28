"""

預測 Pipeline 結果的腳本

"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

# 確保從 scripts/ 資料夾執行可以找到 ml/ 和 app/ 模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.data_loader import load_custom_weather_data
from ml.pv_pipeline import PVModelPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='用已訓練好的模型做預測')
    parser.add_argument('--model', 
                        type=str, 
                        default='models/bias_corrector.pkl', 
                        help='已經訓練好的模型權重檔案路徑，預設為 best_model.pkl')
    parser.add_argument('--input', 
                        type=str, 
                        default='data/processed/processed_weather_data.csv', 
                        help='要輸入作為預測的 CSV 檔案路徑，必須包含 timestamp 欄位')
    parser.add_argument('--output', 
                        type=str, 
                        default='data/predictions/prediction_results.csv', 
                        help='要輸出的 CSV 檔案路徑，預測結果將會寫入這個檔案')
    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = PVModelPipeline(model_path=args.model, use_ml_correction=True)

    # ------------------------------------------------------------------
    # Step 1: 檢查輸入參數
    # ------------------------------------------------------------------
    if not os.path.isfile(args.model):
        print(f"錯誤: 模型檔案 {args.model} 不存在。請確認路徑是否正確。")
        return
    if not os.path.isfile(args.input):
        print(f"錯誤: 輸入的天氣資料檔案 {args.input} 不存在。請確認路徑是否正確。")
        return

    # ------------------------------------------------------------------
    # Step 2: 讀取輸入的天氣資料 CSV 檔案
    # ------------------------------------------------------------------
    weather_df = load_custom_weather_data(args.input)

    # ------------------------------------------------------------------
    # Step 3: 執行預測
    # ------------------------------------------------------------------
    pipeline = PVModelPipeline(model_path=args.model, use_ml_correction=True)
    prediction_results = pipeline.run(weather_df)

    # ------------------------------------------------------------------
    # Step 4: 將預測結果寫入輸出 CSV 檔
    # ------------------------------------------------------------------
    prediction_results.to_csv(args.output)
    print(f"預測完成，結果已寫入 {args.output}")

if __name__ == "__main__":
    main()

    

