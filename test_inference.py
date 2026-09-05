from backend.services.xray_service import run_inference
from pathlib import Path

# Find a test image
test_images = list(Path('dataset/expanded_benchmark/osteoporosis').glob('*'))[:1]
if test_images:
    print(f'Testing inference on: {test_images[0].name}\n')
    result = run_inference(str(test_images[0]), save_artifacts=False)
    print(f'✓ Inference completed successfully!')
    print(f'  Osteoporosis Score: {result["osteoporosis"]["score"]}')
    print(f'  Risk Level: {result["osteoporosis"]["risk_level"]}')
    print(f'  Model Info: {result["xai_status"]}')
else:
    print('No test images found')
