"""M4 analytics (pure Python, no API). Contract: Workflow v2, section 5.2 / 7.3.

Reviewed and corrected by the Lead:
  - regexes now match the real record wording ("HbA1c levels at 7.5%", "BP of 145/95")
  - LDL/HDL are no longer mistaken for total cholesterol; "hr" in "2 hr 30" is no longer a heart rate
  - HbA1c thresholds follow the Part 7.3 range table (>= 7.0 -> above_target)
  - get_trend removes duplicate readings (chunk overlap repeats lines) and ignores empty test names
  - values with no reference rule get status "not_assessed" instead of a false "normal"
  - find_report_patients / report_text_for let the app summarise an uploaded report (e.g. patient 201)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Union

import pandas as pd

_NUM = r"(\d+(?:\.\d+)?)"
_GAP = r"[^\d\n|]{0,25}?"          # a few words between a label and its number (no digits, no table pipes)


def extract_findings(text: str) -> List[Dict[str, Any]]:
    """Regex extraction of common values. Returns [{name, value, unit, raw}]. Only values present in the text."""
    findings: List[Dict[str, Any]] = []
    if not text or not isinstance(text, str):
        return findings

    # Blood pressure -> two findings
    m = re.search(r"\b(?:BP|Blood\s+Pressure)\b" + _GAP + r"(\d{2,3})\s*/\s*(\d{2,3})\b", text, re.I)
    if m:
        findings.append({"name": "Systolic BP", "value": int(m.group(1)), "unit": "mmHg", "raw": m.group(0)})
        findings.append({"name": "Diastolic BP", "value": int(m.group(2)), "unit": "mmHg", "raw": m.group(0)})

    # HbA1c
    m = re.search(r"\b(?:HbA1c|A1c)\b[^\d\n%|]{0,25}?" + _NUM + r"\s*%", text, re.I)
    if m:
        findings.append({"name": "HbA1c", "value": float(m.group(1)), "unit": "%", "raw": m.group(0)})

    # Glucose (fasting / random / plain)
    m = re.search(r"\b(Fasting\s+(?:blood\s+)?glucose|Random\s+(?:blood\s+)?glucose|(?:blood\s+)?glucose)\b"
                  + _GAP + _NUM + r"\s*mg/dL", text, re.I)
    if m:
        kind = m.group(1).lower()
        name = "Fasting Glucose" if "fasting" in kind else ("Random Glucose" if "random" in kind else "Glucose")
        findings.append({"name": name, "value": float(m.group(2)), "unit": "mg/dL", "raw": m.group(0)})

    # Lipids
    for label, name in (("LDL", "LDL Cholesterol"), ("HDL", "HDL Cholesterol")):
        m = re.search(r"\b" + label + r"(?:\s+cholesterol)?\b" + _GAP + _NUM + r"\s*mg/dL", text, re.I)
        if m:
            findings.append({"name": name, "value": float(m.group(1)), "unit": "mg/dL", "raw": m.group(0)})
    for m in re.finditer(r"\b(?:Total\s+Cholesterol|Cholesterol)\b" + _GAP + _NUM + r"\s*mg/dL", text, re.I):
        if re.search(r"(ldl|hdl)\s*$", text[max(0, m.start() - 6):m.start()], re.I):
            continue                                   # "LDL Cholesterol 162" is not the total
        findings.append({"name": "Total Cholesterol", "value": float(re.search(_NUM, m.group(0)).group(1)),
                         "unit": "mg/dL", "raw": m.group(0)})
        break

    # Vitals (HR is matched case-sensitively so "2 hr 30 min" is not a heart rate)
    m = re.search(r"\b(?:Heart\s+Rate|Pulse|(?-i:HR))\b" + _GAP + r"(\d{2,3})\b\s*(?:bpm|beats)?", text, re.I)
    if m:
        findings.append({"name": "Heart Rate", "value": int(m.group(1)), "unit": "bpm", "raw": m.group(0).strip()})
    m = re.search(r"\b(?:SpO2|O2\s+sat(?:uration)?|Oxygen\s+saturation)\b" + _GAP + r"(\d{2,3})\s*%", text, re.I)
    if m:
        findings.append({"name": "SpO2", "value": int(m.group(1)), "unit": "%", "raw": m.group(0)})
    return findings


def flag_abnormal(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adds status (normal|high|low|above_target|not_assessed) and note. Indicative ranges, not clinical guidance."""
    out = []
    for f in findings:
        item = dict(f)
        name, val = item["name"], item["value"]
        status, note = "not_assessed", "No reference range configured; not assessed."

        if name == "Systolic BP":
            status, note = ("high", f"Systolic BP {val} mmHg is elevated (>= 140).") if val >= 140 else ("normal", "Systolic BP within limits.")
        elif name == "Diastolic BP":
            status, note = ("high", f"Diastolic BP {val} mmHg is elevated (>= 90).") if val >= 90 else ("normal", "Diastolic BP within limits.")
        elif name == "HbA1c":
            if val >= 7.0:
                status, note = "above_target", f"HbA1c {val}% is above the usual 7.0% target; indicates sub-optimal glycemic control."
            elif val >= 6.5:
                status, note = "high", f"HbA1c {val}% is in the diabetic range (>= 6.5%)."
            elif val >= 5.7:
                status, note = "high", f"HbA1c {val}% is elevated (5.7-6.4%, prediabetes range)."
            else:
                status, note = "normal", "HbA1c within normal range."
        elif name == "Fasting Glucose":
            if val >= 126:
                status, note = "high", f"Fasting glucose {val} mg/dL is high (>= 126)."
            elif val >= 100:
                status, note = "high", f"Fasting glucose {val} mg/dL is elevated (100-125)."
            elif val < 70:
                status, note = "low", f"Fasting glucose {val} mg/dL is low (< 70)."
            else:
                status, note = "normal", "Fasting glucose within normal range."
        elif name in ("Random Glucose", "Glucose"):
            status, note = ("high", f"Glucose {val} mg/dL is high (>= 200).") if val >= 200 else ("normal", "Glucose below the 200 mg/dL flag level.")
        elif name == "LDL Cholesterol":
            status, note = ("high", f"LDL cholesterol {val} mg/dL is high (>= 130).") if val >= 130 else ("normal", "LDL within limits.")
        elif name == "HDL Cholesterol":
            status, note = ("low", f"HDL cholesterol {val} mg/dL is low (< 40).") if val < 40 else ("normal", "HDL within limits.")
        elif name == "Total Cholesterol":
            status, note = ("high", f"Total cholesterol {val} mg/dL is high (>= 200).") if val >= 200 else ("normal", "Total cholesterol within limits.")
        elif name == "Heart Rate":
            if val > 100:
                status, note = "high", f"Heart rate {val} bpm is high (> 100)."
            elif val < 60:
                status, note = "low", f"Heart rate {val} bpm is low (< 60)."
            else:
                status, note = "normal", "Heart rate within resting range."
        elif name == "SpO2":
            status, note = ("low", f"SpO2 {val}% is low (< 95%).") if val < 95 else ("normal", "SpO2 within limits.")

        item["status"], item["note"] = status, note
        out.append(item)
    return out


_RENAL = ("renal impairment", "kidney disease", "low egfr", "renal insufficiency", "renal failure")
_ARRHYTHMIA = ("severe arrhythmia", "severe cardiovascular arrhythmia", "tachycardia")
NEUTRAL_SAFETY = "No contraindications found in the provided records; verify renal function before Metformin dose changes."


def safety_check(medications: List[str], context_text: str) -> List[str]:
    """Rule-based screening. Pass ONLY the patient's own record text as context_text (never the general
    Section 4 guideline text - that text contains the trigger words for every patient)."""
    meds = [str(m).lower() for m in (medications or [])]
    ctx = (context_text or "").lower()
    warnings = []
    if any("metformin" in m for m in meds) and any(x in ctx for x in _RENAL):
        warnings.append("CRITICAL: Metformin with documented renal impairment/kidney disease in the record - risk of lactic acidosis. Review before continuing.")
    if any(any(x in m for x in ("saba", "albuterol", "salbutamol", "inhaler")) for m in meds) and any(x in ctx for x in _ARRHYTHMIA):
        warnings.append("WARNING: SABA inhaler with documented severe arrhythmia/tachycardia in the record. Monitor heart rate closely.")
    return warnings or [NEUTRAL_SAFETY]


_LINE = re.compile(r"(.*?)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*" + _NUM + r"\s*([A-Za-z/%]+)?")


def get_trend(docs_or_df: Any, test_name: str) -> Union[pd.DataFrame, None]:
    """Repeated readings of one test -> DataFrame(date, value, unit) sorted by date, or None if < 2 points.
    Sources: a DataFrame with columns date/test/value[/unit], or Documents with lines 'Glucose | 2026-09-01 | 142 mg/dL'."""
    name = (test_name or "").strip().lower()
    if not name or docs_or_df is None:
        return None
    points = []
    if isinstance(docs_or_df, pd.DataFrame):
        d = docs_or_df.copy()
        d.columns = [str(c).lower() for c in d.columns]
        if {"date", "test", "value"} <= set(d.columns):
            rows = d[d["test"].astype(str).str.lower().str.contains(name, regex=False, na=False)]
            for _, r in rows.iterrows():
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(r["value"]))
                dt = pd.to_datetime(r["date"], errors="coerce")
                if nums and pd.notna(dt):
                    unit = str(r["unit"]) if "unit" in d.columns and pd.notna(r.get("unit")) else \
                        (re.sub(r"[-+\d.\s]", "", str(r["value"])) or "units")
                    points.append({"date": dt, "value": float(nums[0]), "unit": unit})
    else:
        for doc in docs_or_df:
            for line in str(getattr(doc, "page_content", "")).split("\n"):
                m = _LINE.search(line)
                if m and name in m.group(1).lower():
                    dt = pd.to_datetime(m.group(2), errors="coerce")
                    if pd.notna(dt):
                        points.append({"date": dt, "value": float(m.group(3)), "unit": m.group(4) or "units"})
    if not points:
        return None
    res = pd.DataFrame(points).drop_duplicates(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return res if len(res) >= 2 else None


# ---- uploaded single-patient reports (e.g. data/sample_lab_report.pdf, patient 201) -------------------------
_HDR = re.compile(r"Patient\s*Name\s*:?\s*([^|\n]+?)\s*\|?\s*Patient\s*ID\s*:?\s*(\w+)", re.I)


def report_docs(docs) -> list:
    return [d for d in (docs or []) if getattr(d, "metadata", {}).get("doc_type") == "uploaded_report"]


def find_report_patients(docs) -> List[dict]:
    """[{"patient_id","patient_name","source"}] for uploaded report PDFs whose header names a patient."""
    found, seen = [], set()
    for d in report_docs(docs):
        m = _HDR.search(d.page_content)
        if m and m.group(2) not in seen:
            seen.add(m.group(2))
            found.append({"patient_id": m.group(2), "patient_name": m.group(1).strip(), "source": d.metadata.get("source")})
    return found


def report_text_for(docs, patient_id: str) -> str:
    """Full text of the uploaded report that belongs to patient_id ('' if none). Overlap duplicates removed."""
    hit = next((p for p in find_report_patients(docs) if p["patient_id"] == str(patient_id)), None)
    if not hit:
        return ""
    lines, seen = [], set()
    for d in report_docs(docs):
        if d.metadata.get("source") != hit["source"]:
            continue
        for ln in d.page_content.split("\n"):
            if ln.strip() and ln.strip() not in seen:
                seen.add(ln.strip())
                lines.append(ln.strip())
    return "\n".join(lines)
