from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import eeg, inference

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
