"""
Autonomous Data Scientist Agent — FastAPI entry point.

All endpoints are fully asynchronous:
  POST /api/upload      Upload a CSV dataset
  POST /api/configure   Set analysis configuration
  POST /api/analyze     Run the full async ML pipeline
  POST /api/predict     Predict with the trained best model
  POST /api/chat        Chat with AI assistant
  GET  /api/sessions    List active session IDs
  DELETE /api/sessions/{id}  Delete a session
  GET  /api/health      Health check

Run:
  uvicorn main:app --reload
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any other mpl import

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes import router

app = FastAPI(
    title="Autonomous Data Scientist Agent",
    description=(
        "Upload a dataset — the agent asynchronously cleans, explores, "
        "models, and explains automatically via a REST API."
    ),
    version="2.0.0",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
