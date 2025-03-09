from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import eeg, inference

# Global variable to store the predicted number
predicted_number = None

app = FastAPI(title="EEG Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(eeg.router)
app.include_router(inference.router)

@app.get("/")
async def root():
    return {"message": "EEG Processing API is running"}

@app.get("/prediction")
async def get_prediction():
    """
    Get the current predicted number.
    Returns None if no prediction has been made yet.
    """
    return {"predicted_number": predicted_number}

# Endpoint to update the prediction (can be called by your ML model)
@app.post("/update-prediction")
async def update_prediction(value: int = None):
    """
    Update the predicted number.
    """
    global predicted_number
    predicted_number = value
    return {"status": "success", "predicted_number": predicted_number}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
