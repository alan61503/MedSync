from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
import sys
import uuid
import json

# Ensure repo root is on sys.path so `backend` package imports work when
# running this file directly.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

app = Flask(__name__)
CORS(app, origins=["*"])

BASE_UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


def get_patient_metadata(patient_id: str) -> dict:
    p_dir = BASE_UPLOAD_DIR / patient_id
    info_file = p_dir / "patient_info.json"
    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {
        "id": patient_id,
        "name": f"Patient {patient_id}",
        "age": 52,
        "gender": "Female",
        "medical_history": "Osteoporosis screening study candidate",
    }


def list_images_for_patient(patient_id: str):
    p = BASE_UPLOAD_DIR / patient_id / "xrays"
    if not p.exists():
        return []
    items = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm"):
            items.append({
                "id": str(uuid.uuid4()),
                "filename": f.name,
                "file_path": f"/uploads/{patient_id}/xrays/{f.name}",
                "modality": "XRAY",
            })
    return items


@app.route('/api/patients/summary', methods=['GET'])
def patients_summary():
    summaries = []
    for d in sorted(BASE_UPLOAD_DIR.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        meta = get_patient_metadata(pid)
        xrays_dir = d / "xrays"
        xrays = 0
        if xrays_dir.exists():
            xrays = sum(1 for f in xrays_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.dcm'))
        
        reports_dir = d / "reports"
        reports = 0
        if reports_dir.exists():
            reports = sum(1 for f in reports_dir.iterdir() if f.is_file())

        completion = min(100, max(25, xrays * 25))
        summaries.append({
            "id": pid,
            "name": meta.get("name", f"Patient {pid}"),
            "age": meta.get("age", 50),
            "gender": meta.get("gender", "F"),
            "xrays": xrays,
            "ct": 0,
            "mri": 0,
            "reports": reports,
            "completion": completion,
        })
    return jsonify(summaries)


@app.route('/api/patients', methods=['POST'])
def create_patient():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Patient name is required"}), 400

    # Generate patient ID
    slug = "".join(c.lower() for c in name if c.isalnum()) or "patient"
    patient_id = f"{slug}_{str(uuid.uuid4())[:6]}"
    p_dir = BASE_UPLOAD_DIR / patient_id
    p_dir.mkdir(parents=True, exist_ok=True)
    (p_dir / "xrays").mkdir(exist_ok=True)
    (p_dir / "reports").mkdir(exist_ok=True)

    meta = {
        "id": patient_id,
        "name": name,
        "age": data.get("age") or 45,
        "gender": data.get("gender") or "Female",
        "medical_history": data.get("medical_history", "Osteoporosis research cohort"),
        "symptoms": data.get("symptoms", ""),
    }

    with open(p_dir / "patient_info.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return jsonify(meta), 201


@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    meta = get_patient_metadata(patient_id)
    images = list_images_for_patient(patient_id)
    
    reports_dir = BASE_UPLOAD_DIR / patient_id / "reports"
    reports_list = []
    if reports_dir.exists():
        for rf in sorted(reports_dir.iterdir()):
            if rf.is_file():
                reports_list.append({"filename": rf.name, "path": f"/uploads/{patient_id}/reports/{rf.name}"})

    patient = {
        "id": patient_id,
        "name": meta.get("name", f"Patient {patient_id}"),
        "age": meta.get("age", 50),
        "gender": meta.get("gender", "F"),
        "medical_history": meta.get("medical_history", ""),
        "images": images,
        "reports": reports_list,
    }
    return jsonify(patient)


@app.route('/api/patients/<patient_id>/upload-image', methods=['POST'])
def upload_image(patient_id):
    files = request.files.getlist('files') or []
    saved = []
    target_dir = BASE_UPLOAD_DIR / patient_id / "xrays"
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = target_dir / f.filename
        f.save(dest)
        try:
            from backend.services.xray_service import run_inference
            inf = run_inference(str(dest))
        except Exception as e:
            inf = {"error": str(e)}
        try:
            with open(dest.with_suffix(dest.suffix + '.json'), 'w', encoding='utf-8') as fh:
                json.dump(inf, fh, indent=2)
        except Exception:
            pass
        saved.append({
            "id": str(uuid.uuid4()),
            "filename": f.filename,
            "modality": "XRAY",
            "inference": inf,
            "file_path": f"/uploads/{patient_id}/xrays/{f.filename}"
        })
    return jsonify({"saved": saved})


@app.route('/api/patients/<patient_id>/upload-report', methods=['POST'])
def upload_report(patient_id):
    target_dir = BASE_UPLOAD_DIR / patient_id / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file = request.files.get('file')
    text = request.form.get('text', '')
    
    if file:
        dest = target_dir / file.filename
        file.save(dest)
    elif text:
        dest = target_dir / f"report_{str(uuid.uuid4())[:6]}.txt"
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
    
    return jsonify({"status": "Report uploaded successfully"})


@app.route('/api/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    import shutil
    p_dir = BASE_UPLOAD_DIR / patient_id
    if not p_dir.exists():
        for d in BASE_UPLOAD_DIR.iterdir():
            if d.is_dir() and d.name.lower() == patient_id.lower():
                p_dir = d
                break
    if p_dir.exists() and p_dir.is_dir():
        try:
            shutil.rmtree(str(p_dir), ignore_errors=True)
        except Exception:
            pass
    return jsonify({"status": "Patient deleted successfully", "id": patient_id})



@app.route('/api/patients/<patient_id>/xrays/<filename>', methods=['DELETE'])
def delete_xray(patient_id, filename):
    xray_file = BASE_UPLOAD_DIR / patient_id / "xrays" / filename
    json_file = xray_file.with_suffix(xray_file.suffix + '.json')
    
    deleted = False
    if xray_file.exists():
        try:
            xray_file.unlink()
            deleted = True
        except Exception:
            pass
    if json_file.exists():
        try:
            json_file.unlink()
        except Exception:
            pass
            
    # Also attempt removing associated Grad-CAM heatmaps
    stem = Path(filename).stem
    heatmap_dir = BASE_UPLOAD_DIR / patient_id / "heatmaps"
    if heatmap_dir.exists():
        for hm in heatmap_dir.glob(f"{stem}_*"):
            try:
                hm.unlink()
            except Exception:
                pass

    if deleted:
        return jsonify({"status": "X-ray deleted successfully", "filename": filename})
    return jsonify({"error": "File not found"}), 404


@app.route('/api/patients/<patient_id>/reports/<filename>', methods=['DELETE'])
def delete_report(patient_id, filename):
    report_file = BASE_UPLOAD_DIR / patient_id / "reports" / filename
    if report_file.exists():
        try:
            report_file.unlink()
            return jsonify({"status": "Report deleted successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Report not found"}), 404


@app.route('/api/run-inference', methods=['POST'])
def run_inference_endpoint():
    payload = request.get_json() or {}
    image_url = payload.get('image_url')
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    if isinstance(image_url, str) and image_url.startswith('http'):
        parts = image_url.split('/uploads/')
        if len(parts) > 1:
            rel = parts[1]
            file_path = str(Path(BASE_UPLOAD_DIR) / rel)
        else:
            file_path = image_url
    elif isinstance(image_url, str) and image_url.startswith('/uploads'):
        file_path = str(Path(BASE_UPLOAD_DIR) / image_url.lstrip('/uploads').lstrip('/'))
    else:
        file_path = image_url

    try:
        from backend.services.xray_service import run_inference
        res = run_inference(file_path)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    root = str(BASE_UPLOAD_DIR)
    return send_from_directory(root, filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)


