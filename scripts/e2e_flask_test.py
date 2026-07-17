import requests
from pathlib import Path
import json

BASE = "http://127.0.0.1:8000"
patient_id = "testpatient"
img_path = Path(__file__).resolve().parent.parent / "backend" / "uploads" / patient_id / "xrays" / "sample_xray.png"

print('Image exists:', img_path.exists(), img_path)

# 1) Upload image
files = { 'files': open(img_path, 'rb') }
resp = requests.post(f"{BASE}/api/patients/{patient_id}/upload-image", files=files)
print('\n/upload-image status:', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print('Response not JSON:', resp.text)

# 2) Fetch patient
resp = requests.get(f"{BASE}/api/patients/{patient_id}")
print('\n/patients/{id} status:', resp.status_code)
print(resp.text[:1000])

# 3) Call run-inference directly
image_url = f"http://127.0.0.1:8000/uploads/{patient_id}/xrays/{img_path.name}"
resp = requests.post(f"{BASE}/api/run-inference", json={"image_url": image_url})
print('\n/run-inference status:', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
