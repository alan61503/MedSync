import csv
import json
from pathlib import Path
from typing import List, Tuple

BASE_UPLOAD_DIR = Path(__file__).resolve().parent / 'uploads'

def _collect_inference_data() -> List[Tuple[str, str, float]]:
    """Traverse the uploads directory and collect (patient_id, image_filename, osteoporosis_score).

    Returns a list of tuples suitable for CSV output.
    """
    rows = []
    for patient_dir in BASE_UPLOAD_DIR.iterdir():
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        xrays_dir = patient_dir / 'xrays'
        if not xrays_dir.exists():
            continue
        for json_file in xrays_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                score = data.get('osteoporosis', {}).get('score')
                if score is None:
                    continue
                image_name = json_file.stem
                rows.append((patient_id, image_name, float(score)))
            except Exception:
                continue
    return rows

def generate_ground_truth_csv(output_path: Path) -> Path:
    """Create a CSV file with columns: patient_id,image_filename,true_score.

    The file is written to *output_path* (parent directories are created if needed).
    Returns the path to the created CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _collect_inference_data()
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patient_id', 'image_filename', 'true_score'])
        for row in rows:
            writer.writerow(row)
    return output_path
