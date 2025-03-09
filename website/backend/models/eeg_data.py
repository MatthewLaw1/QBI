from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EEGData(BaseModel):
    """
    Model for EEG data received from Muse 2
    """
    timestamp: float
    channels: List[List[float]]  # [TP9, FP1, FP2, TP10]
    accelerometer: Optional[List[float]] = None
    gyroscope: Optional[List[float]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": 1625097600.123,
                "channels": [
                    [100.5, 101.2, 99.8],  # TP9 channel data points
                    [150.3, 151.1, 149.7],  # FP1 channel data points
                    [145.8, 146.2, 144.9],  # FP2 channel data points
                    [110.2, 111.5, 109.8]   # TP10 channel data points
                ],
                "accelerometer": [0.1, 0.2, 9.8],
                "gyroscope": [0.01, 0.02, 0.01]
            }
        }

class InferenceResult(BaseModel):
    """
    Model for inference results
    """
    prediction: Any
    confidence: float 