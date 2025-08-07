# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analyze_routes import router as analyze_router

app = FastAPI(
    title="Basketball Shot Analysis API",
    description="Processes user videos and returns shooting form feedback",
    version="1.0.0",
)

# Allow mobile app to communicate with backend (for local dev use *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the /analysis routes
app.include_router(analyze_router, prefix="/analysis")

