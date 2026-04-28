from fastapi import APIRouter
from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_one

router = APIRouter()

@router.post('/predict', response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    result = predict_one(payload.dict())
    return PredictionResponse(**result)
