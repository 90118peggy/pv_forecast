from pydantic import BaseModel

class PredictionRequest(BaseModel):
    datetime: str
    temp_air: float
    temp_dew: float
    ghi: float
    dni: float
    dhi: float
    wind_speed: float
    wind_direction: float
    albedo: float
    pressure: float

class PredictionResponse(BaseModel):
    datetime: str
    pvlib_ac: float
    is_daytime: bool | None = None
    ml_correction: float | None = None
    corrected_ac: float | None = None
