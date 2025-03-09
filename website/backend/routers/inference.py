from fastapi import APIRouter, HTTPException
from models.eeg_data import EEGData, InferenceResult
from services.eeg_processor import process_eeg_data

router = APIRouter(prefix="/inference", tags=["Model Inference"])

@router.post("/predict", response_model=InferenceResult)
async def predict(data: EEGData):
    """
    Run inference on processed EEG data
    """
    try:
        # Step 1: Preprocess the EEG data
        processed_data = process_eeg_data(data)
        
        # Step 2: Get prediction from the model
        prediction = get_prediction(processed_data)
        return {"prediction": prediction, "confidence": float(prediction.max())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}") 