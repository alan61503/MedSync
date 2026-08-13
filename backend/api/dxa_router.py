from fastapi import APIRouter, HTTPException, UploadFile, File
from ..services.file_service import save_file
from ..services.dxa_service import run_dxa_bmd

router = APIRouter()

@router.post("/run-dxa-bmd")
def run_dxa_bmd_endpoint(dxa_path: str | None = None, file: UploadFile | None = File(None)):
    """Accept either a backend file path or an uploaded DXA file."""
    if file is not None:
        dxa_path = save_file("standalone", "dxa", file.filename, file.file)
    if not dxa_path:
        raise HTTPException(status_code=400, detail="dxa_path or file required")
    result = run_dxa_bmd(dxa_path)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])  # propagate internal errors
    return result
