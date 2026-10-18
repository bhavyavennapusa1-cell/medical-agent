# Makes `pytest tests/<file>` work on its own: puts the repo root on sys.path so modules import.
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
