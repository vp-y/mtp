import sys
from pathlib import Path

# make `import bus` work even when the package is not pip-installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
