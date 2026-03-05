"""
配置文件

API 配置API的模型路徑、參數
還有NSRDB2的API key配置

"""

Location = {
    'latitude': 35.76,
    'longitude': -91.65,
    'altitude': 81,
    'name': 'U.S.'
}

System = {
    'surface_tilt': 20,
    'surface_azimuth': 180,
    'modules_per_string': 2,
    'strings_per_inverter': 1
}

Module_Parameters = {
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

Inverter_Parameters = {
    'database': 'SandiaInverter',
    'Name': 'SMA_America__SB3000US__240V_',
}

Temperature_Model_Parameters = {
    'model': 'sapm',
    'type': 'open_rack_glass_polymer'
}