# PV Forecast API

這個專案提供一個以 **pvlib 物理模型 + ML bias correction** 為核心的太陽能發電量預測 API。API 目前以 **FastAPI** 實作，主要使用情境是：**每次輸入一筆小時級天氣資料，回傳一筆發電量預測結果**。

## 專案重點

目前的推論流程分成兩層。第一層先用 pvlib 根據場域設定與氣象資料計算 `pvlib_ac`；第二層再用已訓練好的 ML 模型做殘差修正，輸出 `ml_correction` 與 `corrected_ac`。在 inference 端，系統已加入 daytime gate，避免夜間時段被套用 ML 修正。

| 項目 | 說明 |
|---|---|
| API 框架 | FastAPI |
| 物理模型 | pvlib |
| ML 模型 | scikit-learn RandomForestRegressor |
| 輸入 | 單筆小時級 weather JSON |
| 輸出 | `pvlib_ac`、`is_daytime`、`ml_correction`、`corrected_ac` |
| 模型檔 | `models/bias_corrector.pkl` |

## 專案結構

```text
pv_forecast/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── routes/
│   │   └── predictions.py
│   └── services/
│       └── prediction_service.py
├── ml/
│   ├── data_loader.py
│   ├── model_trainer.py
│   └── pv_pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── models/
│   └── bias_corrector.pkl
├── scripts/
│   ├── train_ml_model.py
│   ├── run_prediction.py
│   ├── check_daytime_gate.py
│   └── test_api_hourly.py
├── Dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

### API 啟動方式
#### 本機開發模式

安裝依賴：
```bash
pip install -r requirements.txt
```

再用uvicorn啟動：
```bash
python -m uvicorn app.main:app --reload
```

啟動後可檢查以下端點：
| Endpoint | 說明  |
|---|---|
| GET /  | API 是否啟動  |
| GET /health  | 健康檢查  |
| GET /docs  | Swagger UI  |
| POST /predict| 單筆預測 |

#### Production 啟動方式
正式部署時，建議直接使用：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker 部署
#### 建立映像
```bash
docker build -t pv-forecast-api .
```

#### 啟動容器
```bash
docker run --rm -p 8000:8000 pv-forecast-api
```
若要覆蓋模型路徑或時區設定，可以額外傳入環境變數：
```bash
docker run --rm -p 8000:8000 ^
    -e MODEL_PATH=models/bias_corrector.pkl ^
    -e SITE_TIMEZONE=Etc/GMT+5 ^ 
    pv-forecast-api
```
