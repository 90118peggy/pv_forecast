# PV Forecasting API 

This repository provide solar power generation prediction, according to wheather forecast data and physical data of fields. 

We use PVLid to generate physical features and ML model to do bias correction.

---

# Project Outline

```
pv_forecasting_api/
├── app/                                  # API 服務 (FastAPI/Flask)
│   ├── main.py                           # API 應用入口
│   ├── config.py                         # API 配置 (模型路徑、場域參數等)
│   ├── routes/
│   │   └── predictions.py                # 預測端點
│   └── services/
│       ├── prediction_service.py         # 整合 pvlib 和 ML 模型的預測服務
│       └── schemas.py                    # API 的請求和回應資料結構 (Pydantic)
│
├── ml/                                   # 機器學習核心模組
│   ├── __init__.py
│   ├── pv_pipeline.py                    # 核心預測流程：整合 pvlib 物理預測和 ML 修正
│   ├── model_trainer.py                  # 負責訓練 ML 模型 (學習殘差)
│   ├── data_loader.py                    # 負責載入和預處理天氣數據、歷史發電數據
│   └── utils.py                          # 共用函式庫
│
├── data/                                 # 資料存儲 (原始、處理後)
│   ├── raw/
│   └── processed/
│
├── models/                               # 存放訓練好的模型檔案
│
├── notebooks/                            # 實驗和探索用的 Notebooks
│
├── scripts/                              # 執行腳本
│   ├── train_ml_model.py                 # 執行 ML 模型訓練的腳本
│   ├── run_prediction.py                 # 從命令列執行單次預測的腳本
│   └── start_api.sh                      # 啟動 API 服務的 shell 腳本
│
├── Dockerfile                            # Docker 容器設定檔
├── requirements.txt                      # Python 依賴配置
└── README.md                             # 專案說明文件

```
