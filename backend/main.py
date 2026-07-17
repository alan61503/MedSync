from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .db import engine, Base
from .api import patients
from pathlib import Path

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

# serve uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/api/run-inference")
def run_inference_endpoint(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    # convert URL to file path if it starts with the server host
    if isinstance(image_url, str) and image_url.startswith("http"):
        parts = image_url.split("/uploads/")
        if len(parts) > 1:
            rel = parts[1]
            file_path = str(Path("uploads") / rel)
        else:
            file_path = image_url
    elif isinstance(image_url, str) and image_url.startswith("/uploads"):
        file_path = str(Path(image_url.lstrip("/")))
    else:
        file_path = image_url

    from .services.xray_service import run_inference

    res = run_inference(file_path)
    return res
