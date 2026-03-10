"""
訓練 ML 偏差修正模型的腳本

使用方式：
    # 基本用法（指定訓練數據的時間範圍）
    python scripts/train_ml_model.py --start 2023-01-01 --end 2023-03-07

    # 完整用法（指定所有路徑）
    python scripts/train_ml_model.py \\
        --start 2023-01-01 \\
        --end 2023-03-07 \\
        --weather-path data/processed/processed_weather_data.csv \\
        --actual-path data/raw/10001_hourly_interpolated_2023.csv \\
        --model-output models/bias_corrector.pkl \\
        --test-size 0.2
"""

import argparse
import sys
import os

# 確保從 scripts/ 資料夾執行時，可以找到 ml/ 和 app/ 模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from ml.data_loader import load_custom_weather_data
from ml.pv_pipeline import PVModelPipeline
from ml.model_trainer import MLBiasCorrector


def parse_args():
    parser = argparse.ArgumentParser(
        description='訓練 PV 發電量 ML 偏差修正模型'
    )
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='訓練數據的開始日期，格式：YYYY-MM-DD（例如：2023-01-01）'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='訓練數據的結束日期，格式：YYYY-MM-DD（例如：2023-03-07）'
    )
    parser.add_argument(
        '--weather-path',
        type=str,
        default='data/processed/processed_weather_data.csv',
        help='天氣數據 CSV 的路徑（預設：data/processed/processed_weather_data.csv）'
    )
    parser.add_argument(
        '--actual-path',
        type=str,
        default='data/raw/10001_hourly_interpolated_2023.csv',
        help='實際發電數據 CSV 的路徑（預設：data/raw/10001_hourly_interpolated_2023.csv）'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default='models/bias_corrector.pkl',
        help='訓練完成後模型的儲存路徑（預設：models/bias_corrector.pkl）'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='測試集比例，介於 0.0 到 1.0 之間（預設：0.2）'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PV 發電量 ML 偏差修正模型訓練")
    print("=" * 60)
    print(f"訓練時間範圍：{args.start} ~ {args.end}")
    print(f"天氣數據路徑：{args.weather_path}")
    print(f"實際發電數據路徑：{args.actual_path}")
    print(f"模型輸出路徑：{args.model_output}")
    print(f"測試集比例：{args.test_size}")
    print()

    # ----------------------------------------------------------------
    # Step 1：驗證輸入參數
    # ----------------------------------------------------------------
    try:
        pd.Timestamp(args.start)
        pd.Timestamp(args.end)
    except ValueError:
        print(f"[錯誤] 日期格式不正確，請使用 YYYY-MM-DD 格式")
        sys.exit(1)

    if not os.path.exists(args.weather_path):
        print(f"[錯誤] 找不到天氣數據檔案：{args.weather_path}")
        sys.exit(1)

    if not os.path.exists(args.actual_path):
        print(f"[錯誤] 找不到實際發電數據檔案：{args.actual_path}")
        sys.exit(1)

    if not (0.0 < args.test_size < 1.0):
        print(f"[錯誤] test-size 必須介於 0.0 到 1.0 之間，目前值：{args.test_size}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 2：載入天氣數據並切割訓練時間範圍
    # ----------------------------------------------------------------
    print("[Step 2] 載入天氣數據...")
    weather_df = load_custom_weather_data(args.weather_path)
    weather_train = weather_df[args.start:args.end]

    if weather_train.empty:
        print(f"[錯誤] 在指定的時間範圍 {args.start} ~ {args.end} 內找不到天氣數據")
        sys.exit(1)

    print(f"  天氣數據載入成功，訓練時間範圍內共 {len(weather_train)} 筆")

    # ----------------------------------------------------------------
    # Step 3：執行 pvlib 物理預測（不使用 ML 修正）
    # ----------------------------------------------------------------
    print()
    print("[Step 3] 執行 pvlib 物理預測...")
    pipeline = PVModelPipeline(use_ml_correction=False)
    pvlib_result = pipeline.run(weather_train)

    # pvlib_ac 單位為 W，轉換為 kW 以與實際發電數據一致
    pvlib_kw = pvlib_result['pvlib_ac'] / 1000.0
    print(f"  pvlib 預測完成，最大值：{pvlib_kw.max():.4f} kW，平均值：{pvlib_kw.mean():.4f} kW")

    # ----------------------------------------------------------------
    # Step 4：載入實際發電數據並切割對應時間範圍
    # ----------------------------------------------------------------
    print()
    print("[Step 4] 載入實際發電數據...")
    actual_df = pd.read_csv(args.actual_path, index_col='timestamp', parse_dates=True)

    # 統一時區（與天氣數據保持一致）
    if actual_df.index.tz is None:
        actual_df.index = actual_df.index.tz_localize(weather_df.index.tz)

    actual_kw = actual_df['estimated_avg_power_kw'][args.start:args.end]

    if actual_kw.empty:
        print(f"[錯誤] 在指定的時間範圍 {args.start} ~ {args.end} 內找不到實際發電數據")
        sys.exit(1)

    print(f"  實際發電數據載入成功，共 {len(actual_kw)} 筆")
    print(f"  實際發電最大值：{actual_kw.max():.4f} kW，平均值：{actual_kw.mean():.4f} kW")

    # ----------------------------------------------------------------
    # Step 5：訓練 ML 偏差修正模型
    # ----------------------------------------------------------------
    print()
    print("[Step 5] 訓練 ML 偏差修正模型...")
    corrector = MLBiasCorrector(model_path=args.model_output)
    X, y = corrector.prepare_training_data(weather_train, pvlib_kw, actual_kw)

    print(f"  訓練特徵筆數：{len(X)}")
    print(f"  訓練特徵欄位：{list(X.columns)}")

    metrics = corrector.train(X, y, test_size=args.test_size)

    print()
    print("  訓練結果：")
    print(f"    訓練集 MAE：{metrics['train_mae']:.4f} kW")
    print(f"    訓練集 RMSE：{metrics['train_rmse']:.4f} kW")
    print(f"    測試集 MAE：{metrics['test_mae']:.4f} kW")
    print(f"    測試集 RMSE：{metrics['test_rmse']:.4f} kW")

    # ----------------------------------------------------------------
    # Step 6：儲存模型
    # ----------------------------------------------------------------
    print()
    print("[Step 6] 儲存模型...")
    corrector.save_model()

    print()
    print("=" * 60)
    print("訓練完成！")
    print(f"模型已儲存至：{args.model_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
