"""Medimind - Medical Report Assistant (Streamlit UI).

New visual design (M5) + Lead corrections so the app still matches the contracts in Workflow v2 section 5.2:
  ingest.load_any(file, filename)      -> {"docs", "df", "kind"}
  ingest.list_patients(df)             -> [{"patient_id", "patient_name"}]
  rag.build_index(docs) / rag.ask(question, index, history, df, patient_id)
  summarize.summarize_patient(patient_id, df, index)
  analytics.get_trend(docs_or_df, test) / analytics.find_report_patients(docs)
"""
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
import config  # noqa: E402

st.set_page_config(page_title="Medimind", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
log = logging.getLogger("medimind")

# ----------------------------------------------------------------------------
# Imports: ingest/rag are required; summarize/analytics degrade gracefully
# ----------------------------------------------------------------------------
try:
    from ingest import IngestError, list_patients, load_any
    from rag import ask, build_index
except Exception as exc:  # noqa: BLE001
    log.exception("Core module import failed")
    st.error(f"Core modules ingest.py / rag.py could not be loaded: {exc}")
    st.stop()

MODULE_ERRORS = {}
summarize_patient = None
get_trend = None
find_report_patients = None
try:
    from summarize import summarize_patient
except Exception as exc:  # noqa: BLE001
    MODULE_ERRORS["summarize.py"] = str(exc)
try:
    from analytics import find_report_patients, get_trend
except Exception as exc:  # noqa: BLE001
    MODULE_ERRORS["analytics.py"] = str(exc)

TREND_TESTS = ["Glucose", "HbA1c", "Blood Pressure", "Cholesterol", "Heart Rate"]
SUGGESTED = [
    "What medications were prescribed?",
    "Summarize the key abnormalities.",
    "What is the diagnosis?",
    "What is the blood glucose trend?",
]
STATUS_ICON = {"high": "🔴", "low": "🔴", "critical": "🔴", "above_target": "🟠", "elevated": "🟠", "normal": "🟢"}

_DEFAULTS = {"sources": {}, "docs": [], "df": None, "index": None, "messages": [], "summaries": {}}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


# ----------------------------------------------------------------------------
# Pure helpers (no Streamlit state) - also used by the PDF export
# ----------------------------------------------------------------------------
def txt(x, fallback="Not available"):
    if x is None:
        return fallback
    s = str(x).strip()
    return s if s else fallback


def as_list(value):
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")


def clean_trend(td):
    """Normalise a get_trend() result: datetime/numeric, sorted, NaN rows dropped. Returns None if < 2 points."""
    if td is None or len(td) < 2 or "date" not in td or "value" not in td:
        return None
    x = td.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["date", "value"]).sort_values("date")
    return x if len(x) >= 2 else None


def summary_to_text(s):
    p = s.get("patient") or {}
    lines = [f"Patient: {p.get('name', '?')} (ID {p.get('id', '?')}), age {p.get('age', '?')}",
             f"Diagnosis: {', '.join(map(str, as_list(s.get('diagnosis')))) or 'n/a'}", "", "Key findings:"]
    for f in as_list(s.get("key_findings")):
        lines.append(f"- {f.get('name')}: {f.get('value')} {f.get('unit', '')} [{f.get('status', '')}]"
                     if isinstance(f, dict) else f"- {f}")
    lines += ["", "Abnormalities:"] + [f"- {x}" for x in as_list(s.get("abnormalities"))]
    lines += ["", "Medications:"]
    for m in as_list(s.get("medications")):
        lines.append(f"- {m.get('name')} {m.get('dose', '')} {m.get('frequency', '')}".strip()
                     if isinstance(m, dict) else f"- {m}")
    lines += ["", f"Follow-up: {s.get('follow_up') or 'n/a'}", "", "Safety notes:"]
    lines += [f"- {x}" for x in as_list(s.get("safety_notes"))]
    lines += ["", "Not found:"] + [f"- {x}" for x in as_list(s.get("not_found"))]
    lines += ["", config.DISCLAIMER]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# PDF export (reportlab). trend_getter(test_name) -> DataFrame|None, so it never invents data.
# ----------------------------------------------------------------------------
TEAL = colors.HexColor("#176b87")
INK = colors.HexColor("#17323d")
GREY = colors.HexColor("#66757d")
LINE = colors.HexColor("#d9e3e7")


def chart_drawing(x, test):
    W, H = 175 * mm, 78 * mm
    L, R, T, B = 22 * mm, 8 * mm, 10 * mm, 17 * mm
    pw, ph = W - L - R, H - T - B
    lo, hi = float(x.value.min()), float(x.value.max())
    pad = max((hi - lo) * 0.12, 1)
    lo, hi = lo - pad, hi + pad
    d = Drawing(W, H)
    d.add(String(L, H - 6 * mm, f"{test} trend", fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
    d.add(Line(L, B, L, B + ph, strokeColor=colors.HexColor("#b8c5ca")))
    d.add(Line(L, B, L + pw, B, strokeColor=colors.HexColor("#b8c5ca")))
    for i in range(5):
        y = B + i * ph / 4
        d.add(Line(L, y, L + pw, y, strokeColor=colors.HexColor("#e7ecef"), strokeWidth=0.5))
        d.add(String(L - 12 * mm, y - 2, f"{lo + i * (hi - lo) / 4:.1f}", fontSize=6, fillColor=GREY))
    prev, n = None, len(x)
    for i, (_, r) in enumerate(x.iterrows()):
        xx = L + (i / (n - 1)) * pw
        yy = B + ((float(r.value) - lo) / (hi - lo)) * ph
        if prev:
            d.add(Line(prev[0], prev[1], xx, yy, strokeColor=TEAL, strokeWidth=1.7))
        d.add(Circle(xx, yy, 2.2, strokeColor=TEAL, fillColor=colors.white))
        if n <= 6 or i in (0, n - 1):
            d.add(String(xx - 10, B - 11 * mm, r.date.strftime("%d/%m/%y"), fontSize=6, fillColor=GREY))
        prev = (xx, yy)
    return d


def build_summary_pdf(summary, pid, trend_getter):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"Medimind summary {pid}")
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=19, textColor=TEAL, spaceAfter=5)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, textColor=INK, spaceBefore=9, spaceAfter=5)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#26343b"))
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    head = ParagraphStyle("hc", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    small = ParagraphStyle("s", parent=body, fontSize=7, textColor=GREY)

    def P(s, style=body):
        return Paragraph(esc(txt(s, "")), style)

    def table(rows, widths, header=True):
        data = [[Paragraph(esc(txt(c, "")), head if (header and i == 0) else cell) for c in r]
                for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        style.append(("BACKGROUND", (0, 0), (-1, 0), TEAL) if header
                     else ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5f7")))
        t.setStyle(TableStyle(style))
        return t

    p = summary.get("patient") or {}
    story = [Paragraph("Medimind", title), P("AI-powered Medical Report Assistant"), Spacer(1, 6),
             Paragraph("Patient Information", h),
             table([["Patient ID", txt(p.get("id", pid))], ["Patient Name", txt(p.get("name"))],
                    ["Age", txt(p.get("age"))]], [42 * mm, 125 * mm], header=False),
             Paragraph("Diagnosis", h)]
    story += [P("• " + txt(d)) for d in as_list(summary.get("diagnosis"))] or [P("Not available")]

    story.append(Paragraph("Key Findings", h))
    rows = [["Finding", "Value", "Unit", "Status"]]
    for f in as_list(summary.get("key_findings")):
        if isinstance(f, dict):
            rows.append([txt(f.get("name")), txt(f.get("value")), txt(f.get("unit"), ""),
                         txt(f.get("status"), "normal").replace("_", " ").title()])
    if len(rows) == 1:
        rows.append(["No key findings", "", "", ""])
    story.append(table(rows, [62 * mm, 30 * mm, 25 * mm, 50 * mm]))

    for heading, key in [("Abnormalities", "abnormalities"), ("Safety Notes", "safety_notes"),
                         ("Not Found", "not_found")]:
        story.append(Paragraph(heading, h))
        story += [P("• " + txt(v)) for v in as_list(summary.get(key))] or [P("None reported.")]

    story.append(Paragraph("Medications", h))
    mrows = [["Medication", "Dose", "Frequency"]]
    for m in as_list(summary.get("medications")):
        mrows.append([txt(m.get("name")), txt(m.get("dose"), ""), txt(m.get("frequency"), "")]
                     if isinstance(m, dict) else [txt(m), "", ""])
    if len(mrows) == 1:
        mrows.append(["None reported", "", ""])
    story += [table(mrows, [65 * mm, 35 * mm, 67 * mm]), Paragraph("Follow-up", h), P(summary.get("follow_up")),
              PageBreak(), Paragraph("Lab Trends", h)]

    added = False
    for test in TREND_TESTS:
        td = clean_trend(trend_getter(test))
        if td is None:
            continue
        added = True
        story += [Paragraph(test, h), chart_drawing(td, test), Spacer(1, 4)]
        trows = [["Date", "Value", "Unit"]] + [
            [r.date.strftime("%d/%m/%Y"), txt(r.value), txt(r.get("unit"), "")] for _, r in td.iterrows()]
        story.append(table(trows, [60 * mm, 45 * mm, 50 * mm]))
    if not added:
        story.append(P("Not enough repeated readings available to draw a trend."))
    story += [Spacer(1, 12), Paragraph(esc(config.DISCLAIMER), small)]
    doc.build(story)
    return buf.getvalue()


# ==== UI ====================================================================
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.2rem}
.mm-head{display:flex;align-items:center;gap:12px;margin-bottom:2px}
.mm-logo{width:38px;height:38px;border-radius:10px;background:#176b87;color:white;
display:flex;align-items:center;justify-content:center;font-size:20px}
.mm-title{font-size:28px;font-weight:750}.mm-sub{color:#66757d;font-size:14px;margin:0 0 15px 50px}
.mm-disc{border:1px solid #f0d7a4;background:#fffaf0;color:#76551b;border-radius:8px;
padding:8px 11px;font-size:12px;margin-bottom:16px}
.mm-patient{background:#e9f5f8;border:1px solid #cfe7ed;border-radius:10px;padding:12px 14px;margin-bottom:14px;color:#17323d}
.mm-name{font-weight:700}.mm-meta{color:#4d5f67;font-size:13px;margin-top:3px}
@media(max-width:768px){
.block-container{padding:.8rem .7rem 2rem}
.mm-title{font-size:24px}.mm-sub{margin-left:0}
[data-testid="stHorizontalBlock"]>div{min-width:100%!important;width:100%!important}
.stButton>button{width:100%}
}
</style>
""", unsafe_allow_html=True)


def data_loaded():
    return st.session_state.index is not None


def reset_conversation():
    st.session_state.messages = []
    st.session_state.summaries = {}


def commit_sources(sources):
    """Rebuild docs, df and index from {filename: load_any result}. Raises on failure (state untouched)."""
    docs = [d for res in sources.values() for d in (res.get("docs") or [])]
    dfs = [res["df"] for res in sources.values() if res.get("df") is not None]
    df = None
    if dfs:
        df = (pd.concat(dfs, ignore_index=True).drop_duplicates("patient_id", keep="last").reset_index(drop=True))
    if not docs:
        raise IngestError("No usable medical data was found in the file(s).")
    index = build_index(docs)
    st.session_state.sources = sources
    st.session_state.docs = docs
    st.session_state.df = df
    st.session_state.index = index
    reset_conversation()


def add_files(named_bytes, replace=False):
    """named_bytes = [(filename, bytes)]. ADDS to the loaded data (replace=True starts fresh)."""
    if not config.credentials_configured():
        st.sidebar.error(config.credentials_hint())
        return False
    sources = {} if replace else dict(st.session_state.sources)
    try:
        for name, data in named_bytes:
            sources[name] = load_any(data, name)
        with st.spinner("Reading files and building the search index..."):
            commit_sources(sources)
        return True
    except IngestError as exc:
        st.sidebar.error(f"❌ {exc}")
    except Exception:  # noqa: BLE001
        log.exception("Loading files failed")
        st.sidebar.error("❌ Could not load the data. Check the file and try again.")
    return False


def load_demo():
    files = []
    for name in (config.PATIENT_FILE, config.KB_FILE):
        path = ROOT / config.DATA_DIR / name
        if not path.exists():
            st.sidebar.error(f"Demo file missing: {path.name}")
            return
        files.append((name, path.read_bytes()))
    if add_files(files, replace=True):
        st.sidebar.success("Demo data loaded (3 patients + knowledge base).")


def report_patient_list():
    """Patients that exist only in uploaded report PDFs (header 'Patient Name: .. | Patient ID: ..')."""
    if find_report_patients is None or not st.session_state.docs:
        return []
    known = {p["patient_id"] for p in list_patients(st.session_state.df)} if st.session_state.df is not None else set()
    try:
        return [p for p in find_report_patients(st.session_state.docs) if p["patient_id"] not in known]
    except Exception:  # noqa: BLE001
        log.exception("find_report_patients failed")
        return []


def trend_for(pid, test):
    """Trend for the selected patient. All patients -> every loaded doc. Report patient -> only their report."""
    if get_trend is None:
        return None
    docs = st.session_state.docs
    if pid:
        src = next((p["source"] for p in (find_report_patients(docs) if find_report_patients else [])
                    if p["patient_id"] == str(pid)), None)
        if src is None:
            return None  # spreadsheet patients have no dated series
        docs = [d for d in docs if d.metadata.get("doc_type") == "uploaded_report" and d.metadata.get("source") == src]
    return get_trend(docs, test)


def display_sources(sources):
    if not sources:
        return
    with st.expander("📚 Sources"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {txt(s.get('source'), 'Unknown')}** — page {s.get('page') or 'N/A'}"
                        f" — section: {s.get('section') or 'N/A'} — patient: {s.get('patient_id') or 'general'}")
            if s.get("snippet"):
                st.caption(s["snippet"])


def display_summary(s, pid):
    p = s.get("patient") or {}
    st.markdown(f'<div class="mm-patient"><div class="mm-name">{esc(txt(p.get("name"), "Patient"))}</div>'
                f'<div class="mm-meta">Patient ID: {esc(txt(p.get("id", pid)))} · Age: {esc(txt(p.get("age"), "n/a"))}'
                f'</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🩺 Diagnosis")
    for item in as_list(s.get("diagnosis")) or ["No diagnosis found."]:
        st.write(f"• {item}")

    st.markdown("#### 🔎 Key findings")
    rows = []
    for f in as_list(s.get("key_findings")):
        if isinstance(f, dict):
            status = str(f.get("status") or "normal")
            rows.append({"Test": txt(f.get("name")), "Value": txt(f.get("value")), "Unit": txt(f.get("unit"), ""),
                         "Status": f"{STATUS_ICON.get(status, '⚪')} {status.replace('_', ' ')}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No key findings in the record.")

    for title, key, empty in [("⚠️ Abnormalities", "abnormalities", "None flagged."),
                              ("🛡️ Safety notes", "safety_notes", "No safety notes.")]:
        st.markdown(f"#### {title}")
        for v in as_list(s.get(key)) or [empty]:
            st.write(f"• {txt(v)}")

    st.markdown("#### 💊 Medications")
    meds = as_list(s.get("medications"))
    if meds:
        st.dataframe(pd.DataFrame([
            {"Medication": txt(m.get("name")), "Dose": txt(m.get("dose"), ""), "Frequency": txt(m.get("frequency"), "")}
            if isinstance(m, dict) else {"Medication": txt(m), "Dose": "", "Frequency": ""} for m in meds]),
            hide_index=True, use_container_width=True)
    else:
        st.caption("No medications found.")

    if s.get("follow_up"):
        st.markdown("#### 📅 Follow-up")
        st.write(txt(s.get("follow_up")))
    if as_list(s.get("not_found")):
        st.markdown("#### ❓ Not found in the record")
        for v in as_list(s.get("not_found")):
            st.write(f"• {txt(v)}")
    display_sources(as_list(s.get("sources")))


# ---- header ----------------------------------------------------------------
st.markdown('<div class="mm-head"><div class="mm-logo">✚</div><div class="mm-title">Medimind</div></div>'
            '<div class="mm-sub">AI-powered Medical Report Assistant</div>', unsafe_allow_html=True)
st.markdown(f'<div class="mm-disc">{esc(config.DISCLAIMER)}</div>', unsafe_allow_html=True)
if not config.credentials_configured():
    st.error(config.credentials_hint() + " The assistant cannot call Gemini until this is configured.")
if MODULE_ERRORS:
    st.warning("Some modules are not available: " + "; ".join(f"{k} ({v})" for k, v in MODULE_ERRORS.items()))

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Data")
    if st.button("🧪 Load demo data", key="btn_demo", use_container_width=True):
        load_demo()
    uploads = st.file_uploader("Upload medical reports (PDF, CSV or Excel)", type=["pdf", "csv", "xlsx", "xls"],
                               accept_multiple_files=True, key="uploader")
    if st.button("➕ Add uploaded file(s)", key="btn_add", use_container_width=True, disabled=not uploads):
        if add_files([(f.name, f.getvalue()) for f in uploads]):
            st.success("Added: " + ", ".join(f.name for f in uploads))
    if st.session_state.sources:
        st.markdown("**Loaded files**")
        for name in st.session_state.sources:
            st.caption("• " + name)

    st.divider()
    st.markdown("### 👤 Patient")
    patients = list_patients(st.session_state.df) if st.session_state.df is not None else []
    labels = {"": "All patients"}
    labels.update({p["patient_id"]: f"{p['patient_name']} ({p['patient_id']})" for p in patients})
    labels.update({p["patient_id"]: f"{p['patient_name']} ({p['patient_id']}) - uploaded report"
                   for p in report_patient_list()})
    if len(labels) > 1:
        selected_patient = st.selectbox("Select patient", list(labels), format_func=labels.get,
                                        key="patient_select", label_visibility="collapsed") or None
    else:
        selected_patient = None
        st.caption("Load data to select a patient.")

    st.divider()
    st.markdown("### ⚙️ Actions")
    c_a, c_b = st.columns(2)
    if c_a.button("🗑️ Clear chat", key="btn_clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c_b.button("♻️ Reset data", key="btn_reset", use_container_width=True):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
        st.rerun()

tab_chat, tab_summary, tab_trends = st.tabs(["💬 Chat", "📄 Summary", "📈 Trends"])

# ---- chat ------------------------------------------------------------------
with tab_chat:
    st.markdown("### Ask about the medical records")
    st.caption("Answers come only from the loaded records and always show their sources.")
    if not data_loaded():
        st.info("Click **Load demo data** in the sidebar, or upload a report, to start.")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending_question = q

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                display_sources(msg.get("sources"))

    typed = st.chat_input("Ask a question about the medical records")
    prompt = typed or st.session_state.pop("pending_question", None)
    if prompt:
        if not data_loaded():
            st.warning("Load data first (sidebar).")
        else:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🔍 Reviewing the medical records..."):
                        result = ask(prompt, st.session_state.index, history, st.session_state.df, selected_patient)
                    answer = result.get("answer") or config.NOT_FOUND
                    sources = result.get("sources") or []
                    st.markdown(answer)
                    display_sources(sources)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception:  # noqa: BLE001
                    log.exception("ask() failed")
                    st.error("❌ Unable to generate an answer right now. Please try again.")

# ---- summary ---------------------------------------------------------------
with tab_summary:
    st.markdown("### Medical summary")
    if summarize_patient is None:
        st.warning("Summary is not available: summarize.py could not be loaded.")
    elif not data_loaded():
        st.info("Load the demo data or upload a patient report first.")
    elif selected_patient is None:
        st.info("Select a specific patient in the sidebar to generate a summary.")
    else:
        if st.button("Generate summary", type="primary", key="btn_summary"):
            try:
                with st.spinner("Generating structured summary..."):
                    st.session_state.summaries[selected_patient] = summarize_patient(
                        selected_patient, st.session_state.df, st.session_state.index)
            except Exception:  # noqa: BLE001
                log.exception("summarize_patient failed")
                st.error("❌ Unable to generate the summary right now. Please try again.")
        summary = st.session_state.summaries.get(selected_patient)
        if summary:
            display_summary(summary, selected_patient)
            d1, d2 = st.columns(2)
            d2.download_button("⬇️ Download summary (.txt)", summary_to_text(summary),
                               file_name=f"summary_{selected_patient}.txt", key="dl_txt", use_container_width=True)
            try:
                pdf_bytes = build_summary_pdf(summary, selected_patient, lambda t: trend_for(selected_patient, t))
                d1.download_button("⬇️ Download summary (.pdf)", pdf_bytes, file_name=f"summary_{selected_patient}.pdf",
                                   mime="application/pdf", key="dl_pdf", use_container_width=True)
            except Exception:  # noqa: BLE001
                log.exception("PDF export failed")
                d1.error("The PDF could not be generated. Use the .txt download.")

# ---- trends ----------------------------------------------------------------
with tab_trends:
    st.markdown("### Lab trends")
    st.caption("Repeated dated measurements found in the loaded reports.")
    if get_trend is None:
        st.warning("Trends are not available: analytics.py could not be loaded.")
    elif not data_loaded():
        st.info("Load data or upload a lab report PDF first.")
    else:
        c1, c2 = st.columns(2)
        test = c1.selectbox("Test", TREND_TESTS, key="trend_test")
        custom = c2.text_input("...or type another test name", key="trend_custom").strip()
        test = custom or test
        try:
            td = clean_trend(trend_for(selected_patient, test))
        except Exception:  # noqa: BLE001
            log.exception("trend failed")
            td = None
            st.error("❌ Unable to analyse this trend right now.")
        else:
            if td is None:
                st.info(f"Not enough repeated {test} readings in this report to draw a trend. "
                        "Upload a lab report with several dated values to see a chart.")
            else:
                unit = str(td["unit"].iloc[0]) if "unit" in td.columns else ""
                fig = px.line(td, x="date", y="value", markers=True,
                              labels={"date": "Date", "value": f"{test} ({unit})" if unit else test})
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=35, b=20), plot_bgcolor="white",
                                  paper_bgcolor="white", hovermode="x unified")
                fig.update_traces(line=dict(color="#176b87", width=2.5), marker=dict(size=7, color="#176b87"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with st.expander("Data"):
                    st.dataframe(td, hide_index=True)

st.divider()
st.caption(f"Medimind • decision-support prototype • {datetime.now().year}")
