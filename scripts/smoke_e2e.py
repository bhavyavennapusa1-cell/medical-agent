"""Lead: end-to-end check of M2 + M3 on the real files with real Gemini.
Run:  python scripts/smoke_e2e.py
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import config, ingest, rag

d = ROOT / config.DATA_DIR
p = ingest.load_any(d / config.PATIENT_FILE, config.PATIENT_FILE)
kb = ingest.load_any(d / config.KB_FILE, config.KB_FILE)
idx = rag.build_index(p["docs"] + kb["docs"])
print(f"Indexed {len(p['docs'])} patient docs + {len(kb['docs'])} KB chunks\n")

QUESTIONS = [
    "What medications were prescribed to patient 102?",
    "What is John Doe's diagnosis?",
    "Which patient has asthma?",
    "Who is the oldest patient?",
    "What is Jane's blood glucose trend?",
    "Does Jane have kidney disease?",
    "What is Mary's diagnosis?",
]
for q in QUESTIONS:
    r = rag.ask(q, idx, [], df=p["df"])
    print("Q:", q); print("A:", r["answer"]); print("   sources:", len(r["sources"]), "\n")
