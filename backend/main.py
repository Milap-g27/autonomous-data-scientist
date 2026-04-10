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
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
from uuid import uuid4
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
except ModuleNotFoundError:
    RateLimitExceeded = None
    SlowAPIMiddleware = None

from config import settings
from api.routes import router
from limiter import limiter

app = FastAPI(
    title="Autonomous Data Scientist Agent",
    description=(
        "Upload a dataset — the agent asynchronously cleans, explores, "
        "models, and explains automatically via a REST API."
    ),
    version="2.0.0",
)

logger = logging.getLogger(__name__)

app.state.limiter = limiter

if settings.RATE_LIMIT_ENABLED and SlowAPIMiddleware is not None:
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )
elif settings.RATE_LIMIT_ENABLED:
    logger.warning("slowapi is not installed; running without rate limiting.")

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logging.getLogger("request").info(
        "request_id=%s method=%s path=%s status_code=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response

# ── CORS ── (added LAST so it runs FIRST/outermost, handling OPTIONS before rate limiting)
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
