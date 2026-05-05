import pandas as pd

from app import config
from ml.pv_pipeline import PVModelPipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = PVModelPipeline(
            model_path=config.MODEL_PATH, 
            use_ml_correction=True)
    return _pipeline



def normalize_datetime(datetime_value) -> pd.Timestamp:
    dt = pd.to_datetime(datetime_value)

    if dt.tzinfo is None:
        return dt.tz_localize(config.STIE_TIMEZONE)

    return dt.tz_convert(config.STIE_TIMEZONE)


def build_single_row_weather_df(payload: dict) -> pd.DataFrame:
    dt = normalize_datetime(payload['datetime'])

    df = pd.DataFrame([
        {
            'Year': dt.year,
            'Month': dt.month,
            'Day': dt.day,
            'Hour': dt.hour,
            'Minute': dt.minute,
            'temp_air': payload['temp_air'],
            'temp_dew': payload['temp_dew'],
            'ghi': payload['ghi'],
            'dni': payload['dni'],
            'dhi': payload['dhi'],
            'wind_speed': payload['wind_speed'],
            'wind_direction': payload['wind_direction'],
            'albedo': payload['albedo'],
            'pressure': payload['pressure'],
        }
    ], index=[dt])

    df.index.name = 'datetime'
    return df


def predict_one(payload: dict) -> dict:
    weather_df = build_single_row_weather_df(payload)
    result = get_pipeline().run(weather_df)
    row = result.iloc[0]

    return {
        'datetime': str(result.index[0]),
        'pvlib_ac': float(row['pvlib_ac']),
        'is_daytime': bool(row['is_daytime']) if 'is_daytime' in row else None,
        'ml_correction': float(row['ml_correction']) if 'ml_correction' in row else None,
        'corrected_ac': float(row['corrected_ac']) if 'corrected_ac' in row else None,
    }
