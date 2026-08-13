import os
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from ..services.file_service import save_file
from ..services.ct_bone_service import run_ct_bmd

router = APIRouter()

def _debug_enabled() -> bool:
    """Return True only when DEBUG_CT env var is set to '1'."""
    return os.getenv("DEBUG_CT") == "1"

@router.post("/run-ct-bmd")
def run_ct_bmd_endpoint(ct_path: str | None = None, file: UploadFile | None = File(None)):
    """Accept either a backend file path or an uploaded CT file."""
    if file is not None:
        ct_path = save_file("standalone", "ct", file.filename, file.file)
    if not ct_path:
        raise HTTPException(status_code=400, detail="ct_path or file required")
    result = run_ct_bmd(ct_path)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])  # propagate internal errors
    return result

# --------------------------------------------------------------
# DEBUG ENDPOINT – development only
# --------------------------------------------------------------
@router.post("/run-ct-bmd-debug")
def run_ct_bmd_debug_endpoint(payload: dict, debug: bool = Depends(_debug_enabled)):
    """
    Development‑only endpoint that returns the raw model output and
    intermediate values for troubleshooting.
    """
    if not debug:
        raise HTTPException(status_code=403, detail="Debug endpoint disabled")
    ct_path = payload.get("ct_path")
    if not ct_path:
        raise HTTPException(status_code=400, detail="ct_path required")

    # Run the same pipeline but capture extra data
    from ..services.ct_bone_service import _preprocess_ct, _run_inference
    tensor = _preprocess_ct(ct_path)          # raw input tensor
    raw_bmd = _run_inference(tensor)          # raw (non‑rounded) BMD
    result = run_ct_bmd(ct_path)              # normal result (rounded etc.)

    # Attach debugging information
    result["_debug"] = {
        "model_output_summary": f"tensor shape {list(tensor.shape)}",
        "raw_bmd": raw_bmd,
    }
    return result
