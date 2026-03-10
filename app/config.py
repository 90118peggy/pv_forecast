"""
配置文件

API 配置API的模型路徑、參數
還有NSRDB2的API key配置

"""

# ---- 場域參數 ---- 

SITE_LATITUDE = 39.48
SITE_LONGITUDE = -76.301
SITE_ALTITUDE = 52.22
SITE_NAME = 'U.S.'

# ---- 系統參數 ----
SURFACE_TILT = 20
SURFACE_AZIMUTH = 180
MODULES_PER_STRING = 2
STRINGS_PER_INVERTER = 1

# ---- 逆變器參數 ----
INVERTER_DATABASE = 'SandiaInverter'
INVERTER_NAME = 'SMA_America__SB3000US__240V_'

# ---- 模組參數 ----
MODULE_PARAMETERS = {
    'Name': 'Evergreen ES-A-210-fa2 (Manual)',
    'Technology': 'multi-Si',
    'Bifacial': 0, # 0 表示非雙面模組
    'STC': 210.0,
    'PTC': 189.9, # 規格書上有 PTC 數值，可以一併加入
    'A_c': 1.59,  # 模組面積 (m^2)，從規格書的尺寸 (1.656m x 0.961m) 計算得出
    'Length': 1.656,
    'Width': 0.961,
    'N_s': 72,    # 電池串聯數量，規格書中有提到 (72 cells)
    'I_sc_ref': 8.48,
    'V_oc_ref': 33.7,
    'I_mp_ref': 7.73,
    'V_mp_ref': 27.2,
    'alpha_sc': (0.06 / 100) * 8.48, # 短路電流的絕對溫度係數 (A/°C)
    'beta_oc': (-0.34 / 100) * 33.7,  # 開路電壓的絕對溫度係數 (V/°C)

    'T_NOCT': 45.0, # 規格書中有標明 NOCT 為 45 °C
    # 以下是一些模型需要的進階參數，如果規格書沒有，可以先用 0 或預設值
    'a_ref': 1.5, # 理想因子，如果沒有可以先用一個典型值
    'I_L_ref': 8.48, # 通常約等於 I_sc
    'I_o_ref': 1e-9, # 暗電流，如果沒有可以先用一個典型值
    'R_s': 0.5,      # 串聯電阻，如果沒有可以先用一個典型值
    'R_sh_ref': 300  # 並聯電阻，如果沒有可以先用一個典型值
}

# ---- 溫度模型參數 ----
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
TEMPERATURE_PARAMETERS = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']