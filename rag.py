import os
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_random_exponential

import config


def _get_embeddings():
    """Create the Gemini embedding model using the shared project config (API key)."""
    if not config.credentials_configured():
        raise RuntimeError(config.credentials_hint())
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _build_batch_index(docs, embeddings):
    """Build one FAISS index for a small batch of documents."""
    return FAISS.from_documents(docs, embeddings)


def build_index(docs: List[Document]) -> "FAISS":
    """
    Build an in-memory FAISS index from LangChain Documents.

    The document metadata is preserved so ask() can later return
    source/page/section/patient information.
    """
    if not docs:
        raise ValueError("No documents were provided to build the index.")

    embeddings = _get_embeddings()

    batch_size = 20
    index = None

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        batch_index = _build_batch_index(batch, embeddings)

        if index is None:
            index = batch_index
        else:
            index.merge_from(batch_index)

    return index


# ---------------------------------------------------------------------------
# ask()  (M3, Part 5.2 / 7.2)
# patient-aware retrieval, aggregate questions, follow-ups, guardrails, M4 safety
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a clinical documentation assistant for doctors. Answer ONLY using the CONTEXT below.
Rules:
1. If the answer is not in the CONTEXT, reply exactly: "I could not find this information in the provided reports."
2. Never invent values, doses, dates or diagnoses. Quote numbers exactly as written.
3. If a requested trend has fewer than 2 data points, say so and report the single value you have.
4. Mention the patient name and ID when answering about a patient.
5. Be concise (2-6 sentences) and use bullet points for lists of medications or findings.
6. Do not give treatment recommendations beyond what the records already state.
CONTEXT:
{context}
CHAT HISTORY:
{history}
QUESTION: {question}"""

REWRITE_PROMPT = """Rewrite the follow-up question as ONE standalone question, using the chat history to fill in who or what it refers to. Keep patient names and IDs. Output ONLY the question.
CHAT HISTORY:
{history}
FOLLOW-UP: {question}
STANDALONE QUESTION:"""

# Aggregate questions (Part 7.2 step 4): which/who/all/oldest/youngest/how many
_AGGREGATE_RE = re.compile(
    r"\b(oldest|youngest|how many|all patients|every patient|list (?:all )?patients"
    r"|(?:which|who) (?:patient|patients|one|of the patients)|who (?:has|have|is|are))\b"
)
_MED_Q_RE = re.compile(r"medication|medicine|prescrib|drug|dose|dosage|inhaler|tablet", re.I)
_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})['\u2019]s\b|\bpatient (?:named |called )?([A-Z][a-z]{2,})\b")
_NOT_NAMES = {"what", "who", "where", "when", "how", "why", "which", "that", "there", "here",
              "this", "let", "its", "patient"}
# Medications covered by M4's safety rules (Part 7.3 step 3)
_MED_NAMES = ["Metformin", "SABA", "Albuterol", "Salbutamol"]


def _get_llm():
    """Create the Gemini chat model using the shared project config (API key)."""
    if not config.credentials_configured():
        raise RuntimeError(config.credentials_hint())

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=config.TEMPERATURE)


def _text(response) -> str:
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return (content or "").strip()


def _find_patient(df, question):
    """Use M2's find_patient() when ingest.py is available; otherwise skip."""
    if df is None:
        return None
    try:
        from ingest import find_patient
    except ImportError:
        return None
    return find_patient(df, question)


def _row_text(row):
    """Structured patient row (same wording style as ingest.df_to_documents)."""
    return (
        f"Patient ID: {row['patient_id']}. Name: {row['patient_name']}. "
        f"Age: {row['age']}. Diagnosis: {row['diagnosis']}. "
        f"Clinical notes: {row['notes']}."
    )


def _patient_row_text(df, patient_id):
    rows = df[df["patient_id"].astype(str) == str(patient_id)]
    return None if rows.empty else _row_text(rows.iloc[0])


def _format_history(history):
    """Last 6 messages as plain text."""
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in (history or [])[-6:]]
    return "\n".join(lines) if lines else "(none)"


def _all_docs(index):
    try:
        return list(index.docstore._dict.values())
    except Exception:
        return []


def _standalone_question(llm, question, history):
    """One small LLM call: 'and her dose?' -> 'What is Jane Smith's Metformin dose?'"""
    try:
        rewritten = _text(llm.invoke(
            REWRITE_PROMPT.format(history=_format_history(history), question=question)
        ))
    except Exception:
        return question
    return rewritten or question


def _unknown_name(question, df, index):
    """True if the question names a person (e.g. "Mary's diagnosis") who appears nowhere in the data."""
    docs = _all_docs(index)
    if not docs and df is None:
        return False
    corpus = " ".join(d.page_content for d in docs).lower()
    if df is not None:
        corpus += " " + " ".join(df["patient_name"].astype(str)).lower()
    for m in _NAME_RE.finditer(question):
        name = (m.group(1) or m.group(2)).lower()
        if name not in _NOT_NAMES and name not in corpus:
            return True
    return False


def _safety_notes(question, pid, record_context):
    """M4 safety_check warnings for medication questions about one identified patient.

    Only the patient's own records are passed (not the general Section 4 rules),
    otherwise the rule text itself would trigger a warning for every patient.
    """
    if pid is None or not _MED_Q_RE.search(question):
        return []
    low = record_context.lower()
    meds = [m for m in _MED_NAMES if m.lower() in low]
    if not meds:
        return []
    try:
        from analytics import safety_check
        return list(safety_check(meds, record_context))
    except Exception:
        return []


def _sources(docs):
    return [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "section": d.metadata.get("section"),
            "patient_id": d.metadata.get("patient_id"),
            "snippet": d.page_content[:160],
        }
        for d in docs
    ]


def _is_not_found(answer):
    """True if the model replied with the NOT_FOUND sentence (quotes / missing period tolerated)."""
    plain = answer.strip().strip("\"'").strip()
    return config.NOT_FOUND.rstrip(".").lower() in plain.lower() and len(plain) <= len(config.NOT_FOUND) + 10


def _not_found(pid):
    return {"answer": config.NOT_FOUND, "sources": [], "patient_id": pid}


def ask(
    question: str,
    index,
    history: List[dict],
    df=None,
    patient_id: str | None = None,
) -> dict:
    """
    Return {"answer": str, "sources": [...], "patient_id": str | None}.
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "patient_id": patient_id}

    # 1. Follow-ups: rewrite into a standalone question (one small LLM call).
    llm = None
    if history:
        llm = _get_llm()
        question = _standalone_question(llm, question, history)

    # 2. Aggregate questions: answer from the full patient table, not vector search.
    if df is not None and _AGGREGATE_RE.search(question.lower()) and not _find_patient(df, question):
        llm = llm or _get_llm()
        rows = [_row_text(r) for _, r in df.iterrows()]
        answer = _text(llm.invoke(SYSTEM_PROMPT.format(
            context="\n".join(rows), history=_format_history(history), question=question)))
        if _is_not_found(answer):
            return _not_found(None)
        records = [d for d in _all_docs(index) if d.metadata.get("doc_type") == "patient_record"]
        sources = _sources(records) if records else [
            {"source": "patient table", "page": None, "section": None,
             "patient_id": str(r["patient_id"]), "snippet": _row_text(r)[:160]}
            for _, r in df.iterrows()
        ]
        return {"answer": answer, "sources": sources, "patient_id": None}

    # 3. Patient detection: UI dropdown value first, then M2's find_patient().
    pid = str(patient_id) if patient_id else None
    if pid is None:
        found = _find_patient(df, question)
        if found:
            pid = str(found["patient_id"])

    # 4. Guardrail: a named person who is not in the data -> NOT_FOUND.
    if pid is None and _unknown_name(question, df, index):
        return _not_found(None)

    # 5. Structured row FIRST, then top-k chunks (patient + general when a patient is known).
    context_parts, row = [], None
    if pid is not None and df is not None:
        row = _patient_row_text(df, pid)
        if row:
            context_parts.append(row)

    if pid is not None:
        docs = index.similarity_search(
            question, k=config.TOP_K,
            filter=lambda m: m.get("patient_id") in (pid, None),
            fetch_k=max(50, config.TOP_K * 10),
        )
    else:
        docs = index.similarity_search(question, k=config.TOP_K)
    context_parts.extend(d.page_content for d in docs)

    # 6. Nothing to answer from -> NOT_FOUND, no LLM call.
    if not context_parts:
        return _not_found(pid)

    # 7. Grounded answer.
    llm = llm or _get_llm()
    answer = _text(llm.invoke(SYSTEM_PROMPT.format(
        context="\n\n".join(context_parts),
        history=_format_history(history),
        question=question,
    )))

    # 8. Model said "not found" (any wording around the exact sentence) -> exact NOT_FOUND, no sources.
    if _is_not_found(answer):
        return _not_found(pid)

    # 9. M4 safety warnings for medication answers (only if analytics.py exists).
    record_ctx = "\n".join(
        ([row] if row else []) + [d.page_content for d in docs if d.metadata.get("patient_id") == pid]
    )
    warnings = _safety_notes(question, pid, record_ctx)
    if warnings:
        answer += "\n\nSafety check:\n" + "\n".join(f"- {w}" for w in warnings)

    return {"answer": answer, "sources": _sources(docs), "patient_id": pid}
