from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
print('repo_root:', repo_root)
print('repo_root exists:', repo_root.exists())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
print('sys.path[0]:', sys.path[0])
try:
    import backend.services.xray_service as x
    print('Imported backend.services.xray_service OK')
    print('has run_inference:', hasattr(x, 'run_inference'))
except Exception as e:
    import traceback; traceback.print_exc()
