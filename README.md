# MedSync

Stage 1 scaffold for MedSync: FastAPI backend and Next.js frontend.

Folders:
- backend/: FastAPI app
- frontend/: Next.js app
- services/: placeholder AI services

Run backend:

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Run frontend:

cd frontend
npm install
npm run dev
