MedSync backend (FastAPI)

Run (development):

pip install -r requirements.txt
# Note: installing `torch`/`torchvision` may require platform-specific wheels.
# If you run into issues, install torch following instructions at https://pytorch.org
uvicorn main:app --reload --port 8000

Environment:
- Set `DATABASE_URL` for PostgreSQL, default sqlite used for local dev.
