import os
import pickle
import numpy as np
from models.eeg_data import EEGData

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "eeg_model.pkl")

# Initialize model as None, load on first use
model = None

def load_model():
    """Load the trained model from disk"""
    global model
    if model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return model

def get_prediction(data: EEGData):
    """
    Run inference on processed EEG data
    
    Args:
        data: EEG data to run inference on
        
    Returns:
        Model prediction
    """
    # Ensure model is loaded
    model = load_model()
    
    # Prepare data for model
    # This will depend on your model's expected input format
    channels_data = np.array(data.channels)
    
    # Reshape if needed (example: your model might expect a specific shape)
    # For example, if your model expects (1, 4, time_points):
    model_input = channels_data.reshape(1, channels_data.shape[0], -1)
    
    # Get prediction
    prediction = model.predict(model_input)
    
    return prediction 