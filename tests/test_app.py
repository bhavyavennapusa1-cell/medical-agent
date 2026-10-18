"""UI tests with Streamlit AppTest. No Gemini needed: rag is replaced by a fake module."""
import sys, types
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CALLS = []


def _fake_rag():
    m = types.ModuleType("rag")
    m.build_index = lambda docs: {"n": len(docs)}

    def ask(question, index, history, df=None, patient_id=None):
        CALLS.append({"q": question, "history": history, "pid": patient_id})
        return {"answer": f"FAKE: {question}", "patient_id": patient_id,
                "sources": [{"source": "x.xlsx", "page": None, "section": None, "patient_id": "102", "snippet": "snip"}]}
    m.ask = ask
    return m


@pytest.fixture
def app(monkeypatch):
    CALLS.clear()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "rag", _fake_rag())
    monkeypatch.setitem(sys.modules, "summarize", None)   # simulate M4 not delivered
    monkeypatch.setitem(sys.modules, "analytics", None)
    monkeypatch.delenv("USE_MOCKS", raising=False)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def btn(at, text):
    return [b for b in list(at.sidebar.button) + list(at.button) if text in b.label][0]


def test_starts_without_data_and_warns_about_missing_m4(app):
    app.run()
    assert not app.exception
    assert any("not available yet" in w.value for w in app.warning)


def test_demo_load_uses_real_files_and_patient_dropdown(app):
    app.run()
    btn(app, "Load demo data").click().run()
    assert not app.exception
    df = app.session_state["df"]
    assert list(df["patient_name"]) == ["John Doe", "Jane Smith", "Robert Johnson"]
    assert len(app.session_state["docs"]) == 3 + 6          # 3 patient docs + 6 KB chunks
    sel = app.sidebar.selectbox[0]
    assert len(sel.options) == 4 and sel.options[0] == "All patients"


def test_chat_passes_history_without_current_question(app):
    app.run()
    btn(app, "Load demo data").click().run()
    app.sidebar.selectbox[0].select("102").run()          # Jane Smith
    app.chat_input[0].set_value("What medications were prescribed?").run()
    assert not app.exception
    assert CALLS[0]["history"] == [] and CALLS[0]["pid"] == "102"
    app.chat_input[0].set_value("And her dose?").run()
    assert [m["content"] for m in CALLS[1]["history"]][0] == "What medications were prescribed?"
    assert all(set(m) == {"role", "content"} for m in CALLS[1]["history"])
    assert len(CALLS[1]["history"]) == 2                     # previous Q + A only
    assert len(app.chat_message) == 4


def test_suggested_question_button(app):
    app.run()
    btn(app, "Load demo data").click().run()
    btn(app, "What is the diagnosis").click().run()
    assert CALLS and CALLS[0]["q"] == "What is the diagnosis?"


def test_chat_before_loading_data_is_blocked(app):
    app.run()
    app.chat_input[0].set_value("hello").run()
    assert not CALLS and not app.exception


def test_missing_api_key_shows_error(app, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    app.run()
    btn(app, "Load demo data").click().run()
    assert not app.exception
    assert any("GOOGLE_API_KEY" in e.value for e in app.error) or any("GOOGLE_API_KEY" in e.value for e in app.sidebar.error)
    assert app.session_state["index"] is None


def test_mock_mode_summary_and_trend(app, monkeypatch):
    monkeypatch.setenv("USE_MOCKS", "1")
    app.run()
    btn(app, "Load demo data").click().run()
    app.sidebar.selectbox[0].select("102").run()
    btn(app, "Generate summary").click().run()
    assert not app.exception
    assert "102" in app.session_state["summaries"]
    assert any("Jane Smith" in str(m.value) for m in app.metric)


def test_uploaded_report_patient_in_dropdown_and_trend(app, monkeypatch):
    """Real analytics.py with the fake rag: patient 201 from sample_lab_report.pdf appears and the glucose trend draws."""
    monkeypatch.delitem(sys.modules, "analytics", raising=False)
    from ingest import load_any
    app.run()
    btn(app, "Load demo data").click().run()
    up = load_any(ROOT / "data" / "sample_lab_report.pdf", "sample_lab_report.pdf")
    app.session_state["docs"] = app.session_state["docs"] + up["docs"]
    app.run()
    assert not app.exception
    opts = app.sidebar.selectbox[0].options
    assert len(opts) == 5 and opts[-1].startswith("Terrance Vance (201)")
    assert not any("Not enough repeated Glucose" in i.value for i in app.info)
