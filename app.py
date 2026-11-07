"""Application entry point. Keeps the existing Streamlit dashboard while MVC migration proceeds."""
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
COMPATIBILITY_DIR = PROJECT_ROOT / "framework_ia" / "compatibility"
sys.path.insert(0, str(COMPATIBILITY_DIR))
APP_FILE = COMPATIBILITY_DIR / "app_lab01.py"
if not APP_FILE.exists():
    raise FileNotFoundError(f"Dashboard source not found: {APP_FILE}")
runpy.run_path(str(APP_FILE), run_name="__main__")
