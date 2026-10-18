"""Shared project config. Model names and auth live ONLY here.

Auth: Gemini Developer API key (GOOGLE_API_KEY).
  - Local: put it in a .env file (never commit it).
  - Streamlit Community Cloud: add it under App settings -> Secrets.
"""
import os

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

TEMPERATURE = 0.0
TOP_K = 5
NOT_FOUND = "I could not find this information in the provided reports."
DISCLAIMER = "Decision-support prototype only. Not a substitute for professional medical judgment."
DATA_DIR = "data"
PATIENT_FILE = "medical_reports_csv.xlsx"
KB_FILE = "medical_reports_knowledge_base.pdf"


def credentials_configured() -> bool:
    """True if the Gemini API key is available."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def credentials_hint() -> str:
    return ("GOOGLE_API_KEY is not set. Add it to a local .env file, "
            "or to Secrets in Streamlit Community Cloud.")
