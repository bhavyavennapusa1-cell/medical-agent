"""Integration checks against the two actual supplied datasets."""
from pathlib import Path
import pytest
from ingest import load_any, load_pdf, find_patient, IngestError
DATA = Path(__file__).resolve().parents[1] / "data"

def test_real_excel():
    result = load_any(DATA / "medical_reports_csv.xlsx", "medical_reports_csv.xlsx")
    assert result["kind"] == "tabular"
    assert len(result["df"]) == len(result["docs"]) == 3
    assert result["df"]["patient_id"].tolist() == ["101", "102", "103"]
    assert result["df"]["age"].tolist() == [45, 34, 52]
    assert "140/90" in result["docs"][0].page_content
    assert "Metformin 500mg" in result["docs"][1].page_content
    assert "Inhaler" in result["docs"][2].page_content
    assert find_patient(result["df"], "Does") is None
    assert find_patient(result["df"], "Smith")["patient_id"] == "102"

def test_real_kb():
    result = load_any(DATA / "medical_reports_knowledge_base.pdf", "medical_reports_knowledge_base.pdf")
    assert result["kind"] == "knowledge_base" and result["df"] is None
    docs = result["docs"]
    assert {"101", "102", "103"} <= {d.metadata["patient_id"] for d in docs}
    assert all(d.metadata["doc_type"] == "knowledge_base" for d in docs)
    assert all(d.metadata["page"] in (2, 3) for d in docs)
    assert all(not d.page_content.startswith("2. Patient") for d in docs)
    assert any("renal impairment" in d.page_content for d in docs)
    assert any("Vector Indexing" in d.page_content for d in docs)
    assert all(set(d.metadata) == {"source", "doc_type", "page", "section", "patient_id", "patient_name"} for d in docs)
    for pid in ("101", "102", "103"):
        assert any(d.metadata["patient_id"] == pid and d.metadata["section"].startswith("3.") for d in docs)

def test_real_kb_does_not_index_patient_table():
    docs = load_pdf(DATA / "medical_reports_knowledge_base.pdf", "medical_reports_knowledge_base.pdf", "knowledge_base")
    assert all("Patient Census" not in d.page_content for d in docs)
