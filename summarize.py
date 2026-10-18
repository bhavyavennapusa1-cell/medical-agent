"""M4 structured summary. Contract: Workflow v2, section 5.3.  summarize_patient(patient_id, df, index) -> dict

Reviewed and corrected by the Lead:
  - uses rag._get_llm() so the API-key setup is shared
  - the LLM only extracts medications, follow-up and not_found; patient header, diagnosis, findings, flags and
    safety notes are produced by code (no hallucinated values, no fake "Extracted Meds" placeholder)
  - safety_check sees ONLY the patient's own record (Section 4 guideline text would warn every Metformin patient)
  - retrieval is patient-filtered (patient's section + general chunks), like rag.ask
  - works for uploaded single-patient reports (e.g. sample_lab_report.pdf, patient 201) when the id is not in df
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

import analytics
import config


class _Med(BaseModel):
    name: str
    dose: str = "not specified"
    frequency: str = "not specified"


class _Extract(BaseModel):
    medications: List[_Med] = Field(default_factory=list)
    follow_up: str = ""
    not_found: List[str] = Field(default_factory=list)


PROMPT = """You are a clinical documentation assistant. Extract ONLY what is written in the CONTEXT.
- medications: every medication named in the record, with its dose and frequency exactly as written.
  Use "not specified" when the dose or frequency is not given. Never guess a dose.
- follow_up: the follow-up / monitoring instructions stated in the context (empty string if none).
- not_found: short phrases for important items that are NOT in the context (for example "No time-series glucose values available").
Do not add diagnoses, lab values or advice that are not in the context.

CONTEXT:
{context}
"""


def _not_found_summary(pid) -> Dict[str, Any]:
    return {"patient": {"id": str(pid), "name": "Unknown", "age": None}, "diagnosis": "Not found",
            "key_findings": [], "abnormalities": [], "medications": [], "follow_up": "",
            "safety_notes": [], "not_found": ["Patient not found in the loaded data"], "sources": []}


def _all_docs(index) -> list:
    try:
        return list(index.docstore._dict.values())
    except Exception:  # noqa: BLE001
        return []


_MED_RE = re.compile(r"\b([A-Z][a-z]{3,})\s+(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?))\b")
_FREQ_RE = re.compile(r"\b(once daily|twice daily|three times daily|daily|every \d+ hours?|at bedtime|with (?:the )?\w+ meal|with dinner)\b", re.I)


def _regex_meds(text: str) -> List[dict]:
    """Offline fallback if the LLM call fails: 'Metformin 500mg ... daily' -> medication dict."""
    meds, seen = [], set()
    for m in _MED_RE.finditer(text):
        if m.group(1).lower() in seen:
            continue
        seen.add(m.group(1).lower())
        f = _FREQ_RE.search(text[m.end():m.end() + 60])
        meds.append({"name": m.group(1), "dose": m.group(2).replace(" ", ""), "frequency": f.group(1) if f else "not specified"})
    return meds


def _extract(index_text: str, retries: int = 1) -> Optional[_Extract]:
    try:
        from rag import _get_llm
        structured = _get_llm().with_structured_output(_Extract)
    except Exception:  # noqa: BLE001
        return None
    for _ in range(retries + 1):
        try:
            out = structured.invoke(PROMPT.format(context=index_text))
            if isinstance(out, dict):
                out = _Extract(**out)
            if out is not None:
                return out
        except Exception:  # noqa: BLE001
            continue
    return None


def summarize_patient(patient_id: str, df, index) -> dict:
    pid = str(patient_id)
    docs = _all_docs(index)

    # 1. The patient's own record: spreadsheet row first, otherwise an uploaded report with that id
    record_text, patient, diagnosis, src = "", None, "", None
    if df is not None:
        rows = df[df["patient_id"].astype(str) == pid]
        if not rows.empty:
            r = rows.iloc[0]
            patient = {"id": pid, "name": str(r["patient_name"]), "age": int(r["age"])}
            diagnosis = str(r["diagnosis"])
            record_text = f"Patient ID: {pid}. Name: {r['patient_name']}. Age: {r['age']}. Diagnosis: {diagnosis}. Clinical notes: {r['notes']}."
            rec = next((d for d in docs if d.metadata.get("doc_type") == "patient_record"
                        and str(d.metadata.get("patient_id")) == pid), None)
            src = {"source": rec.metadata.get("source") if rec else config.PATIENT_FILE, "page": None, "section": "Patient record"}
    if patient is None:
        text = analytics.report_text_for(docs, pid)
        if not text:
            return _not_found_summary(pid)
        hit = next(p for p in analytics.find_report_patients(docs) if p["patient_id"] == pid)
        age = re.search(r"\bAge\s*:?\s*(\d{1,3})\b", text, re.I)
        dx = re.search(r"Diagnosis\s*:?\s*(.+)", text, re.I)
        patient = {"id": pid, "name": hit["patient_name"], "age": int(age.group(1)) if age else None}
        diagnosis = dx.group(1).strip() if dx else "Not stated in the report"
        record_text = text
        src = {"source": hit["source"], "page": None, "section": "Uploaded report"}

    # 2. Retrieval: this patient's guideline section + general chunks (never another patient's)
    retrieved, sources = "", [src]
    if index is not None:
        try:
            hits = index.similarity_search(f"{diagnosis} {record_text[:200]}", k=3,
                                           filter=lambda m: m.get("patient_id") in (pid, None) and m.get("doc_type") != "uploaded_report",
                                           fetch_k=50)
            retrieved = "\n".join(h.page_content for h in hits)
            for h in hits:
                s = {"source": h.metadata.get("source"), "page": h.metadata.get("page"), "section": h.metadata.get("section")}
                if s not in sources:
                    sources.append(s)
        except Exception:  # noqa: BLE001
            pass

    # 3. LLM extracts medications / follow-up / not_found; fall back to regex if the API fails
    ext = _extract(record_text + ("\n\nGUIDELINES FOR THIS CONDITION:\n" + retrieved if retrieved else ""))
    not_found: List[str] = []
    if ext is not None:
        meds = [m.model_dump() for m in ext.medications]
        low = (record_text + " " + retrieved).lower()
        meds = [m for m in meds if m["name"].split()[0].lower() in low]       # drop invented medications
        follow_up, not_found = ext.follow_up, list(ext.not_found)
    else:
        meds = _regex_meds(record_text + " " + retrieved)
        follow_up = ""
        not_found.append("AI extraction unavailable; medications taken from the record text by rules only.")

    # 4. Code-generated parts (rules, not the LLM)
    flagged = analytics.flag_abnormal(analytics.extract_findings(record_text))
    abnormalities = [f["note"] for f in flagged if f["status"] in ("high", "low", "above_target")]
    safety = analytics.safety_check([m["name"] for m in meds], record_text) if meds else []

    if "diabet" in diagnosis.lower() and not any("glucose" in f["name"].lower() for f in flagged):
        not_found.append("No time-series glucose values available")
    if not meds:
        not_found.append("No medications documented")
    seen, nf = set(), []
    for x in not_found:
        if x.strip() and x.strip().lower() not in seen:
            seen.add(x.strip().lower()); nf.append(x.strip())

    return {"patient": patient, "diagnosis": diagnosis,
            "key_findings": [{k: f[k] for k in ("name", "value", "unit", "status")} for f in flagged],
            "abnormalities": abnormalities, "medications": meds, "follow_up": follow_up,
            "safety_notes": safety, "not_found": nf, "sources": sources}
