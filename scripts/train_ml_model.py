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
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    parser.add_argument(
        '--split-method',
        type=str,
        choices=['random', 'time', 'both', 'walk-forward'],
        default='both',
        help='訓練/評估切分方式：random、time、both，或 walk-forward'
    )
    parser.add_argument(
        '--save-split',
        type=str,
        choices=['random', 'time'],
        default='time',
        help='當 split-method=both 時，指定要儲存哪一種切分訓練出的模型（預設：time）'
    )
    parser.add_argument(
        '--wf-strategy',
        type=str,
        choices=['expanding', 'rolling'],
        default='rolling',
        help='walk-forward 切分策略（預設：rolling）'
    )
    parser.add_argument(
        '--wf-initial-train-size',
        type=float,
        default=0.5,
        help='walk-forward 初始訓練視窗比例（預設：0.5）'
    )
    parser.add_argument(
        '--wf-test-size',
        type=float,
        default=0.1,
        help='walk-forward 每折測試視窗比例（預設：0.1）'
    )
    parser.add_argument(
        '--wf-step-size',
        type=float,
        default=0.1,
        help='walk-forward 每次前進步長比例（預設：0.1）'
    )
    parser.add_argument(
        '--wf-train-window',
        type=float,
        default=0.5,
        help='rolling 策略訓練窗比例（預設：0.5；expanding 會忽略）'
    )
    parser.add_argument(
        '--wf-max-splits',
        type=int,
        default=5,
        help='walk-forward 最多折數（預設：5）'
    )
    parser.add_argument(
        '--wf-daytime-only',
        action='store_true',
        help='walk-forward 訓練時只使用白天樣本'
    )
    parser.add_argument(
        '--wf-save-train-mode',
        type=str,
        choices=['last-window', 'all-data'],
        default='last-window',
        help='walk-forward 儲存模型時的最終重訓方式（預設：last-window）'
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
    print(f"切分方式：{args.split_method}")
    if args.split_method == 'walk-forward':
        print(
            "walk-forward 設定："
            f"strategy={args.wf_strategy}, "
            f"initial={args.wf_initial_train_size}, "
            f"test={args.wf_test_size}, "
            f"step={args.wf_step_size}, "
            f"train_window={args.wf_train_window}, "
            f"max_splits={args.wf_max_splits}, "
            f"daytime_only={args.wf_daytime_only}, "
            f"save_train_mode={args.wf_save_train_mode}"
        )
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
    base_corrector = MLBiasCorrector(model_path=args.model_output)
    X, y = base_corrector.prepare_training_data(weather_train, pvlib_kw, actual_kw)
    # 這裡需要寫一個Daytime Focus 的資料提取方法，確保訓練資料只包含白天的數據（例如：GHI > 0）
    # X = X[X['GHI'] > 0]
    # y = y[X.index]

    print(f"  訓練特徵筆數：{len(X)}")
    print(f"  訓練特徵欄位：{list(X.columns)}")

    split_methods = ['random', 'time'] if args.split_method == 'both' else [args.split_method]
    trained_models = {}
    metrics_by_split = {}

    if args.split_method == 'walk-forward':
        wf_trainer = MLBiasCorrector(model_path=args.model_output)
        wf_trainer.feature_names = list(base_corrector.feature_names)

        wf_metrics = wf_trainer.evaluate_walk_forward(
            X,
            y,
            initial_train_size=args.wf_initial_train_size,
            test_size=args.wf_test_size,
            step_size=args.wf_step_size,
            strategy=args.wf_strategy,
            train_window=args.wf_train_window,
            daytime_only=args.wf_daytime_only,
            max_splits=args.wf_max_splits,
        )

        print()
        print("  [walk-forward] 各 fold 測試結果：")
        print(
            wf_metrics[
                [
                    'fold',
                    'train_size',
                    'test_size',
                    'test_mae_kW',
                    'test_rmse_kW',
                    'accuracy_percent_peak_norm',
                ]
            ].to_string(
                index=False,
                formatters={
                    'test_mae_kW': '{:.4f}'.format,
                    'test_rmse_kW': '{:.4f}'.format,
                    'accuracy_percent_peak_norm': '{:.2f}'.format,
                },
            )
        )

        print()
        print("  [walk-forward] 平均結果：")
        print(f"    平均 MAE：{wf_metrics['test_mae_kW'].mean():.4f} kW")
        print(f"    平均 RMSE：{wf_metrics['test_rmse_kW'].mean():.4f} kW")
        print(f"    平均 Accuracy：{wf_metrics['accuracy_percent_peak_norm'].mean():.2f}%")

        saver = MLBiasCorrector(model_path=args.model_output)
        saver.feature_names = list(base_corrector.feature_names)

        if args.wf_save_train_mode == 'all-data':
            X_final = X
            y_final = y
            mode_text = 'all-data'
        else:
            wf_splits = saver.generate_walk_forward_splits(
                X,
                y,
                initial_train_size=args.wf_initial_train_size,
                test_size=args.wf_test_size,
                step_size=args.wf_step_size,
                strategy=args.wf_strategy,
                train_window=args.wf_train_window,
                max_splits=args.wf_max_splits,
            )
            X_final, _, y_final, _ = wf_splits[-1]
            mode_text = 'last-window'

        if args.wf_daytime_only:
            day_mask = saver.build_daytime_mask(X_final)
            X_fit = X_final.loc[day_mask]
            y_fit = y_final.loc[day_mask]
        else:
            X_fit = X_final
            y_fit = y_final

        if len(X_fit) == 0:
            print("[錯誤] 儲存模型前可用訓練樣本為 0，請調整 walk-forward 或 daytime-only 參數")
            sys.exit(1)

        saver.model.fit(X_fit, y_fit)
        train_pred = saver.model.predict(X_fit)
        train_mae = mean_absolute_error(y_fit, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_fit, train_pred))
        saver.is_trained = True

        print()
        print("  [walk-forward] 儲存模型前最終重訓：")
        print(f"    重訓模式：{mode_text}")
        print(f"    訓練樣本數：{len(X_fit)}")
        print(f"    訓練集 MAE：{train_mae:.4f} kW")
        print(f"    訓練集 RMSE：{train_rmse:.4f} kW")

        corrector = saver
        selected_split = f"walk-forward ({args.wf_strategy})"
    else:
        for split_method in split_methods:
            trainer = MLBiasCorrector(model_path=args.model_output)
            trainer.feature_names = list(base_corrector.feature_names)

            metrics = trainer.train(X, y, test_size=args.test_size, split_method=split_method)

            trained_models[split_method] = trainer
            metrics_by_split[split_method] = metrics

            print()
            print(f"  [{split_method}] 訓練結果：")
            print(f"    訓練集筆數：{metrics['train_size']}")
            print(f"    測試集筆數：{metrics['test_size']}")
            print(f"    訓練集 MAE：{metrics['train_mae']:.4f} kW")
            print(f"    訓練集 RMSE：{metrics['train_rmse']:.4f} kW")
            print(f"    測試集 MAE：{metrics['test_mae']:.4f} kW")
            print(f"    測試集 RMSE：{metrics['test_rmse']:.4f} kW")

        if args.split_method == 'both':
            print()
            print("  [比較摘要] 兩種切分測試集指標：")
            print(
                "    random -> "
                f"MAE: {metrics_by_split['random']['test_mae']:.4f} kW, "
                f"RMSE: {metrics_by_split['random']['test_rmse']:.4f} kW"
            )
            print(
                "    time   -> "
                f"MAE: {metrics_by_split['time']['test_mae']:.4f} kW, "
                f"RMSE: {metrics_by_split['time']['test_rmse']:.4f} kW"
            )

        selected_split = args.save_split if args.split_method == 'both' else args.split_method
        corrector = trained_models[selected_split]

    # ----------------------------------------------------------------
    # Step 6：儲存模型
    # ----------------------------------------------------------------
    print()
    print("[Step 6] 儲存模型...")
    corrector.save_model()
    print(f"  已儲存切分方式：{selected_split}")

    print()
    print("=" * 60)
    print("訓練完成！")
    print(f"模型已儲存至：{args.model_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
