from fastapi import APIRouter, HTTPException
from models.eeg_data import EEGData, InferenceResult
from services.model_service import get_prediction

router = APIRouter(prefix="/inference", tags=["Model Inference"])

@router.post("/predict", response_model=InferenceResult)
async def predict(data: EEGData):
    """
    Run inference on processed EEG data
    """
    try:
        # Get prediction from the model
        prediction = get_prediction(data)
        return {"prediction": prediction, "confidence": prediction.max()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}") 