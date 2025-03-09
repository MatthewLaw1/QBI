from fastapi import APIRouter, HTTPException, Request
from models.eeg_data import EEGData, InferenceResult
from services.eeg_processor import process_eeg_data
from services.model_predictor import get_prediction

router = APIRouter(prefix="/inference", tags=["Model Inference"])

@router.post("/predict", response_model=InferenceResult)
async def predict(data: EEGData, request: Request):
    """
    Run inference on processed EEG data
    """
    try:
        # Step 1: Preprocess the EEG data
        processed_data = process_eeg_data(data)

        # Step 2: Extract only the last 612 samples from each channel
        for i in range(len(processed_data["processed_channels"])):
            # If the channel has more than 612 samples, take only the last 612
            if len(processed_data["processed_channels"][i]) > 612:
                processed_data["processed_channels"][i] = processed_data["processed_channels"][i][-612:]
        
        # Step 2: Get prediction from the model
        prediction = get_prediction(processed_data)
        
        # Step 3: Update the app state
        # Assuming prediction is a number or can be converted to one
        prediction_value = int(prediction) if hasattr(prediction, "__len__") else int(prediction.max())
        request.app.state.predicted_number = prediction_value
        
        return {"prediction": prediction, "confidence": float(prediction.max())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}") 