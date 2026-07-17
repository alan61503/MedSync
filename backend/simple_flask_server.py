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
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"]) 

BASE_UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


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
    # Return a simple summary list the frontend expects
    summaries = []
    for d in sorted(BASE_UPLOAD_DIR.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        xrays_dir = d / "xrays"
        xrays = 0
        if xrays_dir.exists():
            xrays = sum(1 for f in xrays_dir.iterdir() if f.is_file())
        # placeholder counts for CT/MRI/reports
        ct = 0
        mri = 0
        reports = 0
        completion = min(100, xrays * 10)
        summaries.append({
            "id": pid,
            "name": f"Patient {pid}",
            "xrays": xrays,
            "ct": ct,
            "mri": mri,
            "reports": reports,
            "completion": completion,
        })
    return jsonify(summaries)


@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    # Return a minimal patient object frontend expects
    images = list_images_for_patient(patient_id)
    patient = {
        "id": patient_id,
        "name": f"Patient {patient_id}",
        "age": 50,
        "gender": "F",
        "images": images,
        "reports": [],
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
        # try to run inference
        try:
            from backend.services.xray_service import run_inference
            inf = run_inference(str(dest))
        except Exception as e:
            inf = {"error": str(e)}
        # save inference json
        try:
            with open(dest.with_suffix(dest.suffix + '.json'), 'w', encoding='utf-8') as fh:
                json.dump(inf, fh, indent=2)
        except Exception:
            pass
        saved.append({"id": str(uuid.uuid4()), "filename": f.filename, "modality": "XRAY", "inference": inf, "file_path": f"/uploads/{patient_id}/xrays/{f.filename}"})
    return jsonify({"saved": saved})


@app.route('/api/run-inference', methods=['POST'])
def run_inference_endpoint():
    payload = request.get_json() or {}
    image_url = payload.get('image_url')
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    # convert to local path if starts with /uploads or http://localhost
    if isinstance(image_url, str) and image_url.startswith('http'):
        parts = image_url.split('/uploads/')
        if len(parts) > 1:
            rel = parts[1]
            file_path = str(Path(BASE_UPLOAD_DIR) / rel)
        else:
            file_path = image_url
    elif isinstance(image_url, str) and image_url.startswith('/uploads'):
        file_path = str(Path(image_url.lstrip('/')))
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
    # Serve files from uploads directory
    root = str(BASE_UPLOAD_DIR)
    return send_from_directory(root, filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
