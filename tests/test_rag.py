"""M3 tests for rag.py.

Part A (offline): fake embedder + fake LLM, mock Documents. Runs anywhere.
Part B (real): Part 9 question table with real Gemini + real data files.
               Auto-skipped unless GOOGLE_API_KEY is set and ingest.py + data/ files exist.
Run:  pytest tests/test_rag.py -v
"""
import hashlib, os, re, sys, types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import rag  # noqa: E402

KB = "medical_reports_knowledge_base.pdf"
XL = "medical_reports_csv.xlsx"


# ------------------------------------------------------------------ Part A: offline
class FakeEmb(Embeddings):
    """Offline bag-of-words hash embedder (NOT Gemini)."""
    def _v(self, t):
        v = np.zeros(512, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()
    def embed_documents(self, ts): return [self._v(t) for t in ts]
    def embed_query(self, t): return self._v(t)


class FakeLLM:
    """Answers with .answer; rewrite calls (REWRITE_PROMPT) return .rewrite."""
    def __init__(self, answer="FAKE ANSWER", rewrite=None):
        self.answer, self.rewrite, self.prompts = answer, rewrite, []
    def invoke(self, prompt):
        self.prompts.append(prompt)
        out = self.rewrite if "STANDALONE QUESTION:" in prompt else self.answer
        return types.SimpleNamespace(content=out)
    @property
    def answer_prompts(self): return [p for p in self.prompts if "STANDALONE QUESTION:" not in p]
    @property
    def rewrite_prompts(self): return [p for p in self.prompts if "STANDALONE QUESTION:" in p]


def _meta(doc_type, pid, name, section=None, page=1, source=XL):
    return {"source": source, "doc_type": doc_type, "page": page,
            "section": section, "patient_id": pid, "patient_name": name}


def mock_docs():
    return [
        Document(page_content="Patient ID: 101. Name: John Doe. Age: 45. Diagnosis: Hypertension. Clinical notes: BP 140/90. DASH diet, aerobic exercise 150 min/week. Follow-up in 4 weeks.", metadata=_meta("patient_record", "101", "John Doe")),
        Document(page_content="Patient ID: 102. Name: Jane Smith. Age: 34. Diagnosis: Type 2 Diabetes. Clinical notes: HbA1c levels at 7.5%. Prescribed Metformin 500mg orally once daily with evening meal.", metadata=_meta("patient_record", "102", "Jane Smith")),
        Document(page_content="Patient ID: 103. Name: Robert Johnson. Age: 52. Diagnosis: Asthma. Clinical notes: Mild expiratory wheezing. SABA rescue inhaler for acute episodes. Peak flow twice daily.", metadata=_meta("patient_record", "103", "Robert Johnson")),
        Document(page_content="3.1 Hypertension (Patient ID 101): BP 140/90 indicates elevated blood pressure.", metadata=_meta("knowledge_base", "101", "John Doe", "3.1 Hypertension", 1, KB)),
        Document(page_content="3.2 Type 2 Diabetes Mellitus (Patient ID 102): HbA1c 7.5% is sub-optimal glycemic control; Metformin with daily glucose self-monitoring.", metadata=_meta("knowledge_base", "102", "Jane Smith", "3.2 Type 2 Diabetes Mellitus", 1, KB)),
        Document(page_content="3.3 Asthma (Patient ID 103): mild expiratory wheezing; SABA rescue inhaler, peak flow twice daily.", metadata=_meta("knowledge_base", "103", "Robert Johnson", "3.3 Asthma", 2, KB)),
        Document(page_content="4. Drug safety: cross-check Metformin against renal impairment; SABA inhalers against severe arrhythmia.", metadata=_meta("knowledge_base", None, None, "4 Drug Safety", 2, KB)),
    ]


DF = pd.DataFrame([
    dict(patient_id="101", patient_name="John Doe", age=45, diagnosis="Hypertension", notes="BP 140/90"),
    dict(patient_id="102", patient_name="Jane Smith", age=34, diagnosis="Type 2 Diabetes", notes="HbA1c levels at 7.5%. Prescribed Metformin 500mg daily."),
    dict(patient_id="103", patient_name="Robert Johnson", age=52, diagnosis="Asthma", notes="Mild expiratory wheezing. SABA rescue inhaler."),
])


def _fake_find_patient(df, text):
    """Stand-in for M2's find_patient (whole-word id / first / last name)."""
    for _, r in df.iterrows():
        tokens = [str(r.patient_id)] + r.patient_name.lower().split()
        if any(re.search(rf"\b{re.escape(t)}\b", text.lower()) for t in tokens):
            return r.to_dict()
    return None


def _fake_safety_check(meds, ctx):
    """Stand-in for M4's safety_check (Part 7.3 step 3 rules)."""
    c, out = ctx.lower(), []
    if "metformin" in [m.lower() for m in meds] and ("renal impairment" in c or "kidney disease" in c):
        out.append("Metformin with renal impairment")
    if any(m.lower() in ("saba", "albuterol", "salbutamol") for m in meds) and "severe arrhythmia" in c:
        out.append("SABA with severe arrhythmia")
    return out or ["No contraindications found in the provided records; verify renal function before Metformin dose changes."]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(rag, "_get_embeddings", lambda: FakeEmb())
    monkeypatch.setitem(sys.modules, "ingest", types.SimpleNamespace(find_patient=_fake_find_patient))
    monkeypatch.setitem(sys.modules, "analytics", types.SimpleNamespace(safety_check=_fake_safety_check))
    llm = FakeLLM()
    monkeypatch.setattr(rag, "_get_llm", lambda: llm)
    return rag.build_index(mock_docs()), llm


def test_build_index_and_batching(monkeypatch):
    monkeypatch.setattr(rag, "_get_embeddings", lambda: FakeEmb())
    docs = mock_docs()
    assert rag.build_index(docs).index.ntotal == 7
    assert rag.build_index([docs[i % 7] for i in range(45)]).index.ntotal == 45  # 20+20+5 merged
    with pytest.raises(ValueError):
        rag.build_index([])


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        rag._get_embeddings()
    with pytest.raises(RuntimeError):
        rag._get_llm()


@pytest.mark.parametrize("q,pid,key", [
    ("What medications were prescribed to Jane?", "102", "Metformin"),
    ("What is John Doe's diagnosis?", "101", "Hypertension"),
    ("What was Robert prescribed?", "103", "SABA"),
])
def test_patient_retrieval_contract(env, q, pid, key):
    idx, llm = env
    r = rag.ask(q, idx, [], df=DF)
    assert set(r) == {"answer", "sources", "patient_id"}
    assert r["patient_id"] == pid
    assert r["sources"] and all(set(s) == {"source", "page", "section", "patient_id", "snippet"} for s in r["sources"])
    assert {s["patient_id"] for s in r["sources"]} <= {pid, None}
    assert all(len(s["snippet"]) <= 160 for s in r["sources"])
    ctx = llm.answer_prompts[-1].split("CONTEXT:\n")[1]
    assert ctx.startswith(f"Patient ID: {pid}.") and key in ctx   # structured row first


def test_dropdown_patient_and_no_patient(env):
    idx, _ = env
    assert rag.ask("anything", idx, [], df=DF, patient_id="102")["patient_id"] == "102"
    assert rag.ask("What are the drug safety rules?", idx, [])["patient_id"] is None


@pytest.mark.parametrize("e", ["", "   "])
def test_empty_question(env, e):
    idx, llm = env
    r = rag.ask(e, idx, [])
    assert r["sources"] == [] and r["answer"] == "Please enter a question." and not llm.prompts


def test_aggregate_uses_full_table_without_vector_search(env):
    _, llm = env
    for q in ["Who is the oldest patient?", "Which patient has asthma?", "How many patients are there?"]:
        r = rag.ask(q, None, [], df=DF)          # index=None -> proves no vector search
        ctx = llm.answer_prompts[-1].split("CONTEXT:\n")[1].split("\nCHAT HISTORY")[0]
        assert all(n in ctx for n in ["John Doe", "Jane Smith", "Robert Johnson"])
        assert r["patient_id"] is None and r["sources"]


def test_named_patient_with_which_is_not_aggregate(env):
    idx, llm = env
    r = rag.ask("Which medications were prescribed to Jane?", idx, [], df=DF)
    assert r["patient_id"] == "102"
    assert "John Doe" not in llm.answer_prompts[-1].split("CONTEXT:\n")[1].split("CHAT HISTORY")[0].split("\n\n")[0]


def test_followup_rewrite(env):
    idx, llm = env
    llm.rewrite = "What is Jane Smith's Metformin dose?"
    hist = [{"role": "user", "content": "What medications was Jane prescribed?"},
            {"role": "assistant", "content": "Metformin."}]
    r = rag.ask("And her dose?", idx, hist, df=DF)
    assert len(llm.rewrite_prompts) == 1
    assert r["patient_id"] == "102"                    # found via rewritten question
    assert "QUESTION: What is Jane Smith's Metformin dose?" in llm.answer_prompts[-1]


def test_no_rewrite_call_without_history_and_history_trimmed(env):
    idx, llm = env
    rag.ask("Jane dose", idx, [], df=DF)
    assert not llm.rewrite_prompts
    llm.rewrite = "Jane dose"
    hist = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    rag.ask("Jane dose", idx, hist, df=DF)
    h = llm.rewrite_prompts[-1].split("CHAT HISTORY:\n")[1].split("\nFOLLOW-UP")[0]
    assert h.count("user:") == 6 and "m9" in h and "m3" not in h


@pytest.mark.parametrize("model_says", [
    config.NOT_FOUND, f'"{config.NOT_FOUND}"', config.NOT_FOUND.rstrip(".")])
def test_model_not_found_is_normalised(env, model_says):
    idx, llm = env
    llm.answer = model_says
    r = rag.ask("Does Jane have kidney disease?", idx, [], df=DF)
    assert r["answer"] == config.NOT_FOUND and r["sources"] == [] and r["patient_id"] == "102"


def test_unknown_patient_name_skips_llm(env):
    idx, llm = env
    r = rag.ask("What is Mary's diagnosis?", idx, [], df=DF)
    assert r["answer"] == config.NOT_FOUND and r["patient_id"] is None and not llm.prompts


def test_whats_and_drug_possessives_are_not_unknown_names(env):
    idx, llm = env
    assert rag.ask("What's the diagnosis for Jane?", idx, [], df=DF)["answer"] == "FAKE ANSWER"
    assert rag.ask("What are Metformin's risks?", idx, [])["answer"] == "FAKE ANSWER"


def test_safety_warning_only_for_medication_questions_with_patient(env):
    idx, llm = env
    r = rag.ask("What medications were prescribed to Jane?", idx, [], df=DF)
    assert "Safety check:" in r["answer"] and "verify renal function" in r["answer"]
    # general Section 4 rule text must NOT trigger a patient warning
    assert "renal impairment" not in r["answer"].split("Safety check:")[1].split("verify")[0]
    assert "Safety check:" not in rag.ask("What is Jane's diagnosis?", idx, [], df=DF)["answer"]
    assert "Safety check:" not in rag.ask("What medications exist?", idx, [])["answer"]


def test_safety_triggers_when_patient_record_mentions_renal(env, monkeypatch):
    idx, llm = env
    df = DF.copy(); df.loc[1, "notes"] += " Kidney disease noted."
    r = rag.ask("What medications were prescribed to Jane?", idx, [], df=df)
    assert "Metformin with renal impairment" in r["answer"]


def test_works_without_m4_analytics(env, monkeypatch):
    idx, _ = env
    monkeypatch.setitem(sys.modules, "analytics", None)   # simulates ImportError
    r = rag.ask("What medications were prescribed to Jane?", idx, [], df=DF)
    assert r["answer"] == "FAKE ANSWER"


# ------------------------------------------------------------------ Part B: real Gemini + real files
DATA = ROOT / config.DATA_DIR
REAL = bool(os.getenv("GOOGLE_API_KEY")) and (DATA / XL).exists() and (DATA / KB).exists() and (ROOT / "ingest.py").exists()
real = pytest.mark.skipif(not REAL, reason="needs GOOGLE_API_KEY, data/ files and M2's ingest.py")

PART9 = [  # (question, patient_id arg, [facts that must appear, case-insensitive])
    ("What medications were prescribed to patient 102?", None, ["metformin", "500"]),
    ("What is John Doe's diagnosis?", None, ["hypertension", "140/90"]),
    ("Which patient has asthma?", None, ["robert"]),
    ("Who is the oldest patient?", None, ["robert", "52"]),
    ("What is Jane's blood glucose trend?", None, ["7.5"]),
    ("What medications was Robert prescribed?", None, ["saba"]),
    ("What monitoring was advised for Robert?", None, ["peak flow"]),
    ("Summarize key abnormalities for Jane", None, ["7.5"]),
    ("Summarize key abnormalities for John", None, ["140/90"]),
    ("Any medication safety concerns for Jane?", None, ["metformin"]),
]
UNANSWERABLE = ["Does Jane have kidney disease?", "What is Mary's diagnosis?", "What is John's cholesterol?"]


@pytest.fixture(scope="module")
def real_env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import ingest
    t = ingest.load_any(str(DATA / XL), XL)
    kb = ingest.load_any(str(DATA / KB), KB)
    return rag.build_index(t["docs"] + kb["docs"]), t["df"]


@real
@pytest.mark.parametrize("q,pid,facts", PART9)
def test_part9_answerable(real_env, q, pid, facts):
    idx, df = real_env
    r = rag.ask(q, idx, [], df=df, patient_id=pid)
    assert set(r) == {"answer", "sources", "patient_id"}
    for f in facts:
        assert f in r["answer"].lower(), f"{f!r} missing in: {r['answer']}"


@real
@pytest.mark.parametrize("q", UNANSWERABLE)
def test_part9_unanswerable(real_env, q):
    idx, df = real_env
    assert config.NOT_FOUND in rag.ask(q, idx, [], df=df)["answer"]


@real
def test_part9_followup(real_env):
    idx, df = real_env
    hist = [{"role": "user", "content": "What medications were prescribed to Jane Smith?"},
            {"role": "assistant", "content": "Jane Smith (ID 102) was prescribed Metformin 500mg once daily."}]
    r = rag.ask("And her dose?", idx, hist, df=df)
    assert "500" in r["answer"] and r["patient_id"] == "102"
