from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .db import engine, Base
from .api import patients
from .api import ct_router, dxa_router
from pathlib import Path
from .services.file_service import BASE_UPLOAD_DIR

from fastapi import HTTPException

app = FastAPI(title="MedSync API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # create tables
    Base.metadata.create_all(bind=engine)


app.include_router(patients.router, prefix="/api")
app.include_router(ct_router.router, prefix="/api")
app.include_router(dxa_router.router, prefix="/api")

# serve uploaded files
app.mount("/uploads", StaticFiles(directory=str(BASE_UPLOAD_DIR)), name="uploads")





@app.post("/api/ai-verify")
def ai_verify_endpoint(payload: dict):
    """2-Factor AI Verification: send inference results to Groq LLaMA 3.3 70B for clinical validation."""
    inference = payload.get("inference")
    if not inference:
        raise HTTPException(status_code=400, detail="inference data required")

    from .services.llm_service import verify_inference

    result = verify_inference(inference)
    return result
