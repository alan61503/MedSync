from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os

from .. import models, schemas
from ..db import get_db, engine
from ..services.file_service import save_file, detect_modality, is_dicom_file, BASE_UPLOAD_DIR
from ..services.report_service import extract_text_from_pdf
from ..services.xray_service import run_inference
import json
import shutil
from pathlib import Path

router = APIRouter()


@router.post("/patients", response_model=schemas.PatientOut)
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
        inference = run_inference(path)

        # persist inference next to the image and move heatmap into uploads patient folder
        try:
            img_path = Path(path)
            inf_obj = inference or {}
            # move heatmap into uploads/{patient_id}/heatmaps
            heatmap_src = inf_obj.get("heatmap_path")
            if heatmap_src:
                heatmap_src_path = Path(heatmap_src)
                dest_dir = BASE_UPLOAD_DIR / patient_id / "heatmaps"
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / heatmap_src_path.name
                try:
                    shutil.move(str(heatmap_src_path), str(dest_path))
                    inf_obj["heatmap_url"] = f"/uploads/{patient_id}/heatmaps/{dest_path.name}"
                except Exception:
                    inf_obj["heatmap_url"] = None

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
        saved.append({"id": img.id, "filename": img.filename, "modality": img.modality, "inference": inference})

    return {"saved": saved}


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


@router.get("/patients/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.get(models.Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/patients", response_model=List[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).all()


@router.get("/patients/summary")
def patients_summary(db: Session = Depends(get_db)):
    results = []
    patients = db.query(models.Patient).all()
    for p in patients:
        total_images = len(p.images)
        xrays = sum(1 for i in p.images if (i.modality or "").upper() not in ("CT", "MR"))
        ct = sum(1 for i in p.images if (i.modality or "").upper() == "CT")
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
                "mri": mri,
                "reports": total_reports,
                "completion": completion,
            }
        )

    return results


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
        data["images"][key].append({"id": img.id, "filename": img.filename, "path": img.file_path})

    for r in patient.reports:
        data["reports"].append({"id": r.id, "filename": r.filename, "text": (r.text[:500] + "...") if r.text and len(r.text) > 500 else r.text})

    return data
