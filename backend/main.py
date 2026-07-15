from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .db import engine, Base
from .api import patients

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
