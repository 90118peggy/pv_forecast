"""Run hourly API predictions and plot the results."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_COLUMNS = [
    'datetime',
    'temp_air',
    'temp_dew',
    'ghi',
    'dni',
    'dhi',
    'wind_speed',
    'wind_direction',
    'albedo',
    'pressure',
]


def parse_args():
    parser = argparse.ArgumentParser(description='逐小時呼叫 /predict API 並畫出預測曲線')
    parser.add_argument(
        '--url',
        type=str,
        default='http://127.0.0.1:8000/predict',
        help='API 端點 URL',
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'processed_weather_data.csv'),
        help='含有逐小時天氣資料的 CSV 檔案路徑',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=24,
        help='最多送出幾筆資料，預設 24 筆',
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'predictions' / 'api_hourly_results.csv'),
        help='預測結果輸出的 CSV 路徑',
    )
    parser.add_argument(
        '--output-plot',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'predictions' / 'api_hourly_curve.png'),
        help='曲線圖輸出的 PNG 路徑',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='只存圖，不跳出視窗',
    )
    return parser.parse_args()


def load_input_data(input_path: str, limit: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f'輸入檔缺少必要欄位: {missing}')

    df = df.head(limit).copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def build_payload(row: pd.Series) -> dict:
    return {
        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        'temp_air': float(row['temp_air']),
        'temp_dew': float(row['temp_dew']),
        'ghi': float(row['ghi']),
        'dni': float(row['dni']),
        'dhi': float(row['dhi']),
        'wind_speed': float(row['wind_speed']),
        'wind_direction': float(row['wind_direction']),
        'albedo': float(row['albedo']),
        'pressure': float(row['pressure']),
    }


def call_api(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def plot_results(results: pd.DataFrame, output_plot: str, show_plot: bool) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(results['datetime'], results['pvlib_ac'], label='pvlib_ac', linewidth=2)

    if 'corrected_ac' in results.columns and results['corrected_ac'].notna().any():
        plt.plot(results['datetime'], results['corrected_ac'], label='corrected_ac', linewidth=2)

    plt.xlabel('datetime')
    plt.ylabel('prediction')
    plt.title('Hourly PV Prediction from API')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()
    input_df = load_input_data(args.input, args.limit)

    results = []
    for index, row in input_df.iterrows():
        payload = build_payload(row)
        try:
            response_data = call_api(args.url, payload)
            print(
                f"[{index}] {payload['datetime']} -> "
                f"pvlib_ac={response_data.get('pvlib_ac')} "
                f"corrected_ac={response_data.get('corrected_ac')}"
            )
            results.append({**payload, **response_data})
        except requests.RequestException as error:
            print(f"[{index}] {payload['datetime']} 呼叫失敗: {error}")
            results.append({**payload, 'error': str(error)})

    results_df = pd.DataFrame(results)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f'結果已輸出到 {output_csv}')

    valid_results = results_df.dropna(subset=['pvlib_ac']).copy()
    if valid_results.empty:
        print('沒有可繪圖的成功結果。')
        return

    valid_results['datetime'] = pd.to_datetime(valid_results['datetime'])
    plot_results(valid_results, args.output_plot, not args.no_show)
    print(f'曲線圖已輸出到 {args.output_plot}')


if __name__ == '__main__':
    main()
