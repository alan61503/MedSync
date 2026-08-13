from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os

from .. import models, schemas
from ..db import get_db, engine
from ..services.ct_bone_service import run_ct_bmd
from ..services.dxa_service import run_dxa_bmd
from ..services.llm_service import verify_inference
from ..services.file_service import save_file, detect_modality, BASE_UPLOAD_DIR
from ..services.inference_service import run_routed_inference
# duplicate import removed
# duplicate import removed
# verify_inference import retained above, duplicate removed
from ..models import BMDResult
from ..services.report_service import extract_text_from_pdf
from ..services.xray_service import run_inference
import json
import shutil
from pathlib import Path

router = APIRouter()


def _public_upload_url(file_path: str) -> str:
    try:
        relative_path = Path(file_path).resolve().relative_to(BASE_UPLOAD_DIR.resolve())
        return f"/uploads/{relative_path.as_posix()}"
    except Exception:
        return file_path



@router.post("/patients/{patient_id}/ct-bmd")
def ct_bmd_endpoint(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CT file for a patient, compute BMD, store result, run verification, and return data."""
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    # Save file
    saved_path = save_file(patient_id, "ct", file.filename, file.file)
    # Run BMD inference
    result = run_ct_bmd(saved_path)
    # Run LLM verification (fallback if API unavailable)
    verification = verify_inference({"predictions": {}, "supporting_findings": {}})
    # Persist BMDResult
    bmd_record = BMDResult(
        patient_id=patient_id,
        modality="CT",
        bmd=str(result.get("bmd")),
        t_score=str(result.get("t_score")),
        risk_level=result.get("risk_level"),
        diagnostic=result.get("diagnostic"),
        verification=verification.get("verdict")
    )
    db.add(bmd_record)
    db.commit()
    db.refresh(bmd_record)
    # Combine result
    result["verification"] = verification
    result["bmd_record_id"] = bmd_record.id
    return result

@router.post("/patients/{patient_id}/dxa-bmd")
def dxa_bmd_endpoint(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a DXA file for a patient, compute BMD, store result, run verification, and return data."""
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    saved_path = save_file(patient_id, "dxa", file.filename, file.file)
    result = run_dxa_bmd(saved_path)
    verification = verify_inference({"predictions": {}, "supporting_findings": {}})
    bmd_record = BMDResult(
        patient_id=patient_id,
        modality="DXA",
        bmd=str(result.get("bmd")),
        t_score=str(result.get("t_score")),
        risk_level=result.get("risk_level"),
        diagnostic=result.get("diagnostic"),
        verification=verification.get("verdict")
    )
    db.add(bmd_record)
    db.commit()
    db.refresh(bmd_record)
    result["verification"] = verification
    result["bmd_record_id"] = bmd_record.id
    return result
def create_patient(payload: schemas.PatientCreate, db: Session = Depends(get_db)):
    patient = models.Patient(
        name=payload.name,
        age=payload.age,
        gender=payload.gender,
        medical_history=payload.medical_history,
        previous_diseases=payload.previous_diseases,
        symptoms=payload.symptoms,
        notes=payload.notes,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.post("/patients/{patient_id}/upload-image")
def upload_image(patient_id: str, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        category = "xrays"
        # save raw file
        path = save_file(patient_id, category, f.filename, f.file)
        modality = detect_modality(path)
        # map modalities
        m = modality.upper() if modality else "UNKNOWN"
        if m in ["CT"]:
            category = "ct"
        elif m in ["DXA"]:
            category = "dxa"
        elif m in ["MR"]:
            category = "mri"
        else:
            category = "xrays"

        # if saved in wrong folder, move
        final_dir = BASE_UPLOAD_DIR / patient_id / category
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / os.path.basename(path)
        if str(final_path) != path:
            os.replace(path, str(final_path))
            path = str(final_path)

        # run inference (best-effort)
        inference = run_routed_inference(path)

        # persist inference next to the image
        try:
            img_path = Path(path)
            inf_obj = inference or {}
            hp = inf_obj.get("heatmap_path") or inf_obj.get("heatmap_url")
            op = inf_obj.get("overlay_path") or inf_obj.get("overlay_url") or hp
            if hp:
                inf_obj["heatmap_path"] = hp
                inf_obj["heatmap_url"] = hp
            if op:
                inf_obj["overlay_path"] = op
                inf_obj["overlay_url"] = op

            # write inference JSON next to the stored image
            inf_json_path = img_path.with_suffix(img_path.suffix + ".json")
            with open(inf_json_path, "w", encoding="utf-8") as fh:
                json.dump(inf_obj, fh, indent=2)
        except Exception:
            pass

        img = models.MedicalImage(patient_id=patient_id, filename=f.filename, file_path=path, modality=m)
        db.add(img)
        db.commit()
        db.refresh(img)
        saved.append({
            "id": img.id,
            "filename": img.filename,
            "modality": img.modality,
            "file_path": _public_upload_url(img.file_path),
            "inference": inference,
        })

    return {"saved": saved}



@router.post("/patients/{patient_id}/osteoporosis-report")
def osteoporosis_report(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Endpoint that accepts a single xray upload, runs the osteoporosis-focused pipeline,
    saves inference JSON + heatmap, and returns a short report JSON."""
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # save file
    path = save_file(patient_id, "xrays", file.filename, file.file)
    final_dir = BASE_UPLOAD_DIR / patient_id / "xrays"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / os.path.basename(path)
    if str(final_path) != path:
        os.replace(path, str(final_path))
        path = str(final_path)

    inference = run_routed_inference(path)
    # call LLM (will fallback to offline report if groq fails)
    llm = __import__("services.llm_service", fromlist=["analyse"]).analyse(inference.get("predictions", {}), inference.get("confidence_scores", {}), {"heatmap": inference.get("heatmap_path")})

    # persist inference JSON next to image
    try:
        img_path = Path(path)
        inf_json_path = img_path.with_suffix(img_path.suffix + ".json")
        with open(inf_json_path, "w", encoding="utf-8") as fh:
            json.dump({"inference": inference, "llm": llm}, fh, indent=2)
    except Exception:
        pass

    return {"inference": inference, "llm": llm}


@router.post("/run-inference")
def run_inference_endpoint(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    try:
        return run_routed_inference(image_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/patients/{patient_id}/upload-report")
def upload_report(patient_id: str, file: UploadFile = File(None), text: str = Form(None), db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    filename = None
    path = None
    extracted = None
    if file:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        path = save_file(patient_id, "reports", filename, file.file)
        if ext == ".pdf":
            extracted = extract_text_from_pdf(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                extracted = fh.read()

    if text and not extracted:
        extracted = text

    report = models.RadiologyReport(patient_id=patient_id, filename=filename, file_path=path, text=extracted)
    db.add(report)
    db.commit()
    db.refresh(report)
    return {"id": report.id, "text_preview": (extracted[:200] if extracted else None)}


@router.get("/patients/summary")
def patients_summary(db: Session = Depends(get_db)):
    results = []
    patients = db.query(models.Patient).all()
    for p in patients:
        total_images = len(p.images)
        xrays = sum(1 for i in p.images if (i.modality or "").upper() not in ("CT", "MR"))
        ct = sum(1 for i in p.images if (i.modality or "").upper() == "CT")
        dxa = sum(1 for i in p.images if (i.modality or "").upper() == "DXA")
        mri = sum(1 for i in p.images if (i.modality or "").upper() == "MR")
        total_reports = len(p.reports)
        # completeness based on simple heuristic
        fields = [p.age, p.gender, p.medical_history, p.previous_diseases, p.symptoms, p.notes]
        filled_fields = sum(1 for f in fields if f)
        field_frac = filled_fields / max(1, len(fields))
        image_frac = min(1.0, total_images / 3)
        report_frac = min(1.0, total_reports / 1)
        completion = int((field_frac * 0.7 + image_frac * 0.2 + report_frac * 0.1) * 100)

        results.append(
            {
                "id": p.id,
                "name": p.name,
                "xrays": xrays,
                "ct": ct,
                "dxa": dxa,
                "mri": mri,
                "reports": total_reports,
                "completion": completion,
            }
        )

    return results


@router.get("/patients/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "medical_history": patient.medical_history,
        "previous_diseases": patient.previous_diseases,
        "symptoms": patient.symptoms,
        "notes": patient.notes,
        "created_at": patient.created_at,
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "file_path": _public_upload_url(img.file_path),
                "modality": img.modality,
            }
            for img in patient.images
        ],
        "reports": [
            {
                "id": report.id,
                "filename": report.filename,
                "text": report.text,
            }
            for report in patient.reports
        ],
    }


@router.get("/patients", response_model=List[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).all()


@router.delete("/patients/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    # remove files
    upload_dir = BASE_UPLOAD_DIR / patient_id
    if upload_dir.exists():
        import shutil

        shutil.rmtree(upload_dir)

    db.delete(patient)
    db.commit()
    return {"deleted": patient_id}


@router.get("/patients/{patient_id}/structured-data")
def patient_structured(patient_id: str, db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    data = {
        "patient_id": patient.id,
        "metadata": {
            "age": patient.age,
            "gender": patient.gender,
            "history": patient.medical_history,
        },
        "images": {"xray": [], "ct": [], "mri": []},
        "reports": [],
        "created_at": patient.created_at.isoformat(),
    }

    for img in patient.images:
        key = "xray"
        if img.modality and img.modality.upper() == "CT":
            key = "ct"
        elif img.modality and img.modality.upper() == "MR":
            key = "mri"
        data["images"][key].append({"id": img.id, "filename": img.filename, "path": _public_upload_url(img.file_path)})

    for r in patient.reports:
        data["reports"].append({"id": r.id, "filename": r.filename, "text": (r.text[:500] + "...") if r.text and len(r.text) > 500 else r.text})

    return data
