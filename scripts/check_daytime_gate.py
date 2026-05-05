import requests

URL = "http://127.0.0.1:8000/predict"

samples = [
    {
        "datetime": "2023-07-01 00:00:00",
        "temp_air": 24.0,
        "temp_dew": 20.0,
        "ghi": 0.0,
        "dni": 0.0,
        "dhi": 0.0,
        "wind_speed": 1.5,
        "wind_direction": 180.0,
        "albedo": 0.2,
        "pressure": 1013.0,
    },
    {
        "datetime": "2023-07-01 12:00:00",
        "temp_air": 30.0,
        "temp_dew": 24.0,
        "ghi": 900.0,
        "dni": 750.0,
        "dhi": 120.0,
        "wind_speed": 2.0,
        "wind_direction": 180.0,
        "albedo": 0.2,
        "pressure": 1013.0,
    },
    {
        "datetime": "2023-07-01 23:00:00",
        "temp_air": 25.0,
        "temp_dew": 21.0,
        "ghi": 0.0,
        "dni": 0.0,
        "dhi": 0.0,
        "wind_speed": 1.2,
        "wind_direction": 170.0,
        "albedo": 0.2,
        "pressure": 1013.0,
    },
]

for sample in samples:
    response = requests.post(URL, json=sample, timeout=30 )
    response.raise_for_status()
    result = response.json()
    print(sample["datetime"], result)
