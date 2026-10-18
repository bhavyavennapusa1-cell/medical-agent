"""UI-development mocks for the M4 modules (summarize.py / analytics.py).
Only used when env var USE_MOCKS=1 AND the real module is missing. Never used in production.
Shapes follow Workflow v2, sections 5.2 / 5.3.
"""
import pandas as pd


def mock_summarize_patient(patient_id, df, index):
    rows = df[df["patient_id"].astype(str) == str(patient_id)]
    if rows.empty:
        return {"patient": {"id": str(patient_id), "name": "Unknown", "age": None}, "diagnosis": "Not found",
                "key_findings": [], "abnormalities": [], "medications": [], "follow_up": "",
                "safety_notes": [], "not_found": ["Patient not found"], "sources": []}
    r = rows.iloc[0]
    return {
        "patient": {"id": str(r["patient_id"]), "name": r["patient_name"], "age": int(r["age"])},
        "diagnosis": r["diagnosis"],
        "key_findings": [{"name": "HbA1c", "value": 7.5, "unit": "%", "status": "above_target"}],
        "abnormalities": ["MOCK: HbA1c 7.5% indicates sub-optimal glycemic control"],
        "medications": [{"name": "Metformin", "dose": "500mg", "frequency": "once daily"}],
        "follow_up": "MOCK follow-up text",
        "safety_notes": ["MOCK: verify renal function before Metformin dose changes"],
        "not_found": ["No time-series glucose values available"],
        "sources": [{"source": "mock", "page": 1, "section": "3.2"}],
    }


def mock_get_trend(docs_or_df, test_name):
    if str(test_name).lower() != "glucose":
        return None
    return pd.DataFrame({"date": ["2026-08-01", "2026-08-15", "2026-09-01", "2026-09-15", "2026-10-01"],
                         "value": [110, 118, 126, 139, 148], "unit": "mg/dL"})
