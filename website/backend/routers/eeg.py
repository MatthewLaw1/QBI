from fastapi import APIRouter, HTTPException
from models.eeg_data import EEGData
from services.eeg_processor import process_eeg_data

router = APIRouter(prefix="/eeg", tags=["EEG Data"])

@router.post("/receive")
async def receive_eeg_data(data: EEGData):
    """
    Receive EEG data packets from Muse 2 headset
    """
    try:
        # Process the incoming EEG data
        processed_data = process_eeg_data(data)
        return {"status": "success", "processed_data": processed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing EEG data: {str(e)}") 