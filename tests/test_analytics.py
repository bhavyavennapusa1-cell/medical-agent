"""M4 analytics tests (Part 7.3 step 8) on the real data files and the generated sample report."""
import io
from pathlib import Path

import pandas as pd
import pytest

import analytics
from ingest import load_any

DATA = Path(__file__).resolve().parents[1] / "data"


def _flag(text):
    return {f["name"]: f for f in analytics.flag_abnormal(analytics.extract_findings(text))}


def test_blood_pressure_140_90_flagged_high():
    f = _flag("Patient exhibits elevated blood pressure 140/90.")
    assert f["Systolic BP"]["value"] == 140 and f["Systolic BP"]["status"] == "high"
    assert f["Diastolic BP"]["status"] == "high"
    assert _flag("Patient has a high BP of 145/95 mmHg.")["Systolic BP"]["status"] == "high"


def test_hba1c_real_wording_flagged_above_target():
    f = _flag("HbA1c levels at 7.5%. Prescribed Metformin 500mg daily.")
    assert f["HbA1c"]["value"] == 7.5 and f["HbA1c"]["status"] == "above_target"
    assert _flag("HbA1c: 5.2 %")["HbA1c"]["status"] == "normal"


def test_no_invented_findings():
    assert analytics.extract_findings("Mild wheezing observed. Inhaler prescribed for acute episodes.") == []
    assert analytics.extract_findings("") == [] and analytics.extract_findings(None) == []
    assert "Heart Rate" not in _flag("Walked 2 hr 30 minutes")


def test_ldl_is_not_total_cholesterol():
    f = _flag("LDL Cholesterol 162 mg/dL. HDL: 38 mg/dL. Total Cholesterol: 245 mg/dL")
    assert f["LDL Cholesterol"]["value"] == 162 and f["Total Cholesterol"]["value"] == 245
    assert f["HDL Cholesterol"]["status"] == "low"
    assert "Total Cholesterol" not in _flag("LDL Cholesterol 162 mg/dL")


def test_safety_rules():
    assert "CRITICAL" in analytics.safety_check(["Metformin 500mg"], "Patient has severe kidney disease.")[0]
    assert "WARNING" in analytics.safety_check(["SABA rescue inhaler"], "history of severe arrhythmia")[0]
    assert analytics.safety_check(["Metformin"], "HbA1c 7.5%") == [analytics.NEUTRAL_SAFETY]


def test_trend_from_dataframe_and_none_cases():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "test": ["Glucose", "Glucose"], "value": ["120 mg/dL", "140 mg/dL"]})
    t = analytics.get_trend(df, "Glucose")
    assert len(t) == 2 and list(t.columns) == ["date", "value", "unit"] and t["unit"].iloc[0] == "mg/dL"
    assert analytics.get_trend(df, "") is None and analytics.get_trend(df.iloc[:1], "Glucose") is None


def test_trend_on_real_patient_xlsx_is_none():
    r = load_any(DATA / "medical_reports_csv.xlsx", "medical_reports_csv.xlsx")
    assert analytics.get_trend(r["df"], "Glucose") is None
    assert analytics.get_trend(r["docs"], "Glucose") is None


def test_trend_on_sample_report_returns_5_points():
    r = load_any(DATA / "sample_lab_report.pdf", "sample_lab_report.pdf")
    t = analytics.get_trend(r["docs"], "Glucose")
    assert len(t) == 5 and t["value"].is_monotonic_increasing and t["unit"].iloc[0] == "mg/dL"
    assert analytics.get_trend(r["docs"], "HbA1c") is None


def test_sample_report_patient_is_found():
    docs = load_any(DATA / "sample_lab_report.pdf", "sample_lab_report.pdf")["docs"]
    assert analytics.find_report_patients(docs)[0]["patient_id"] == "201"
    text = analytics.report_text_for(docs, "201")
    assert "Metformin 1000mg" in text and text.count("2026-09-01") == 1
