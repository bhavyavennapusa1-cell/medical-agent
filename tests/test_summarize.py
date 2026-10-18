"""summarize_patient tests: real files, fake embedder + fake LLM (no Gemini needed)."""
import sys, types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import rag, summarize  # noqa: E402
from ingest import load_any  # noqa: E402
from test_rag import FakeEmb  # noqa: E402

DATA = ROOT / "data"
XL, KB, SAMPLE = "medical_reports_csv.xlsx", "medical_reports_knowledge_base.pdf", "sample_lab_report.pdf"


class FakeStructured:
    def __init__(self, result): self.result, self.prompts = result, []
    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.result, Exception): raise self.result
        return self.result


class FakeLLM:
    def __init__(self, result): self.s = FakeStructured(result)
    def with_structured_output(self, schema): return self.s


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(rag, "_get_embeddings", lambda: FakeEmb())
    t, kb, up = (load_any(DATA / XL, XL), load_any(DATA / KB, KB), load_any(DATA / SAMPLE, SAMPLE))
    index = rag.build_index(t["docs"] + kb["docs"] + up["docs"])
    return index, t["df"]


def use(monkeypatch, result):
    llm = FakeLLM(result)
    monkeypatch.setattr(rag, "_get_llm", lambda: llm)
    return llm


def test_jane_summary_matches_contract(env, monkeypatch):
    index, df = env
    use(monkeypatch, summarize._Extract(
        medications=[summarize._Med(name="Metformin", dose="500mg", frequency="once daily with evening meal"),
                     summarize._Med(name="Insulin")],               # invented -> must be dropped
        follow_up="Daily glucose self-monitoring", not_found=[]))
    s = summarize.summarize_patient("102", df, index)
    assert set(s) == {"patient", "diagnosis", "key_findings", "abnormalities", "medications", "follow_up",
                      "safety_notes", "not_found", "sources"}
    assert s["patient"] == {"id": "102", "name": "Jane Smith", "age": 34}
    assert s["diagnosis"] == "Type 2 Diabetes"
    assert s["key_findings"] == [{"name": "HbA1c", "value": 7.5, "unit": "%", "status": "above_target"}]
    assert [m["name"] for m in s["medications"]] == ["Metformin"]
    assert s["abnormalities"] and "7.5" in s["abnormalities"][0]
    assert s["safety_notes"] == [summarize.analytics.NEUTRAL_SAFETY]     # NOT a false renal warning from Section 4
    assert "No time-series glucose values available" in s["not_found"]
    assert s["sources"][0]["source"] == XL and all(set(x) == {"source", "page", "section"} for x in s["sources"])


def test_retrieval_never_mixes_other_patients(env, monkeypatch):
    index, df = env
    llm = use(monkeypatch, summarize._Extract())
    summarize.summarize_patient("102", df, index)
    ctx = llm.s.prompts[-1]
    assert "Jane Smith" in ctx and "John Doe" not in ctx and "Robert Johnson" not in ctx and "Terrance" not in ctx


def test_john_bp_and_robert_no_findings(env, monkeypatch):
    index, df = env
    use(monkeypatch, summarize._Extract())
    j = summarize.summarize_patient("101", df, index)
    assert {f["name"] for f in j["key_findings"]} == {"Systolic BP", "Diastolic BP"} and len(j["abnormalities"]) == 2
    assert j["safety_notes"] == [] and "No medications documented" in j["not_found"]
    r = summarize.summarize_patient("103", df, index)
    assert r["key_findings"] == [] and r["patient"]["name"] == "Robert Johnson"


def test_llm_failure_falls_back_to_rules_without_fake_meds(env, monkeypatch):
    index, df = env
    use(monkeypatch, RuntimeError("quota"))
    s = summarize.summarize_patient("102", df, index)
    assert s["medications"] and s["medications"][0]["name"] == "Metformin" and s["medications"][0]["dose"] == "500mg"
    assert all(m["name"] != "Extracted Meds" for m in s["medications"])
    assert any("unavailable" in x for x in s["not_found"])


def test_uploaded_report_patient_201_gets_renal_warning(env, monkeypatch):
    index, df = env
    use(monkeypatch, summarize._Extract(medications=[summarize._Med(name="Metformin", dose="1000mg", frequency="once daily with dinner")],
                                        follow_up="Follow-up in 4 weeks"))
    s = summarize.summarize_patient("201", df, index)
    assert s["patient"] == {"id": "201", "name": "Terrance Vance", "age": 61}
    assert "Type 2 Diabetes" in s["diagnosis"]
    names = {f["name"]: f["status"] for f in s["key_findings"]}
    assert names["HbA1c"] == "above_target" and names["Systolic BP"] == "high" and names["LDL Cholesterol"] == "high"
    assert s["safety_notes"] and "CRITICAL" in s["safety_notes"][0]
    assert s["sources"][0]["source"] == SAMPLE


def test_works_with_no_dataframe_and_unknown_patient(env, monkeypatch):
    index, _ = env
    use(monkeypatch, summarize._Extract())
    assert summarize.summarize_patient("201", None, index)["patient"]["name"] == "Terrance Vance"
    nf = summarize.summarize_patient("999", None, index)
    assert nf["diagnosis"] == "Not found" and nf["not_found"]
