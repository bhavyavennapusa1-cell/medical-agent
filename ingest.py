"""M2 ingestion module. Contract: Hackathon Workflow v2, section 5.2."""
from __future__ import annotations
import io
import re
from pathlib import Path
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

REQUIRED = ('patient_id', 'patient_name', 'age', 'diagnosis', 'notes')
META_KEYS = ('source', 'doc_type', 'page', 'section', 'patient_id', 'patient_name')

class IngestError(Exception):
    """A file cannot be read or does not meet the ingestion contract."""


def _bytes(file):
    if isinstance(file, (str, Path)):
        try: data = Path(file).read_bytes()
        except OSError as exc: raise IngestError(f'Cannot read file: {exc}') from exc
    elif isinstance(file, bytes): data = file
    elif hasattr(file, 'read'):
        try:
            if hasattr(file, 'seek'): file.seek(0)
            data = file.read()
        except Exception as exc: raise IngestError(f'Cannot read uploaded file: {exc}') from exc
    else: raise IngestError('Expected a path, bytes, or binary file object')
    if not data: raise IngestError('Empty file is not supported')
    return io.BytesIO(data)


def _meta(source, doc_type, page=None, section=None, patient_id=None, patient_name=None):
    return dict(zip(META_KEYS, (source, doc_type, page, section, patient_id, patient_name)))


def _normalize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower())).strip('_') for c in df.columns]
    df = df.dropna(how='all')
    if df.empty: raise IngestError('No patient rows found')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise IngestError('Missing required columns: ' + ', '.join(missing))
    for c in ('patient_id', 'patient_name', 'diagnosis', 'notes'):
        df[c] = df[c].fillna('').astype(str).str.strip()
    df['patient_id'] = df['patient_id'].str.replace(r'\.0$', '', regex=True)
    if df['patient_id'].eq('').any() or df['patient_name'].eq('').any():
        raise IngestError('Patient ID and patient name cannot be blank')
    ages = pd.to_numeric(df['age'], errors='coerce')
    if ages.isna().any() or ((ages % 1) != 0).any() or ((ages < 0) | (ages > 125)).any():
        raise IngestError('Age must be a valid whole number from 0 to 125')
    df['age'] = ages.astype(int)
    return df.reset_index(drop=True)


def load_tabular(file, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in ('.csv', '.xlsx', '.xls'): raise IngestError('Tabular file must be CSV or Excel')
    data = _bytes(file)
    try:
        if ext == '.csv':
            df = pd.read_csv(data, dtype={'patient_id': str})
            return _normalize_columns(df)
        engine = 'xlrd' if ext == '.xls' else 'openpyxl'
        book = pd.ExcelFile(data, engine=engine)
        for sheet in book.sheet_names:
            df = pd.read_excel(book, sheet_name=sheet, dtype={'patient_id': str})
            if not df.dropna(how='all').empty: return _normalize_columns(df)
        raise IngestError('No non-empty Excel sheet found')
    except IngestError: raise
    except Exception as exc: raise IngestError(f'Unable to parse {filename}: {exc}') from exc


def df_to_documents(df, source: str) -> list[Document]:
    df = _normalize_columns(df)
    docs = []
    for _, row in df.iterrows():
        content = (f"Patient ID: {row['patient_id']}. Name: {row['patient_name']}. "
                   f"Age: {row['age']}. Diagnosis: {row['diagnosis']}. "
                   f"Clinical notes: {row['notes']}.")
        docs.append(Document(page_content=content, metadata=_meta(source, 'patient_record',
                             patient_id=row['patient_id'], patient_name=row['patient_name'])))
    return docs


# Recognize the numbered knowledge-base sections without including the truncated table.
_HEADING = re.compile(r'^\s*(?:Section\s+)?([3-6](?:\.\d+)?)(?:[.\s:–-]+)(\S.*)?$', re.I)
_PATIENT_ID = re.compile(r'\bPatient\s*ID\s*[:#-]?\s*(10[1-3])\b', re.I)
_SECTION_IDS = {'3.1': '101', '3.2': '102', '3.3': '103'}
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def _pdf_pages(file):
    try:
        with pdfplumber.open(_bytes(file)) as pdf:
            return [(n, page.extract_text() or '', page.extract_tables() or [])
                    for n, page in enumerate(pdf.pages, 1)]
    except IngestError: raise
    except Exception as exc: raise IngestError(f'Unable to parse PDF: {exc}') from exc


def _table_lines(tables):
    lines = []
    for table in tables:
        for row in table:
            if row:
                cells = [re.sub(r'\s+', ' ', str(c or '')).strip() for c in row]
                if any(cells): lines.append(' | '.join(cells))
    return lines


def _chunks(text, source, kind, page, section=None, patient_id=None):
    return [Document(page_content=part, metadata=_meta(source, kind, page, section, patient_id))
            for part in _SPLITTER.split_text(text) if part.strip()]


def load_pdf(file, filename: str, doc_type: str) -> list[Document]:
    if doc_type not in ('knowledge_base', 'uploaded_report'):
        raise IngestError('PDF doc_type must be knowledge_base or uploaded_report')
    pages = _pdf_pages(file)
    docs = []
    if doc_type == 'uploaded_report':
        for page, text, tables in pages:
            content = text.strip()
            table_text = '\n'.join(_table_lines(tables))
            if table_text and table_text not in content: content += '\n' + table_text
            if content.strip(): docs.extend(_chunks(content.strip(), filename, doc_type, page))
    else:
        # KB: collect sections 3.x, 4 and 5 only. Do not index preceding truncated table.
        sections = []
        current = None
        for page, text, _ in pages:
            if page > 1: current = None  # never merge text across a page header
            for line in text.splitlines():
                match = _HEADING.match(line)
                if match:
                    number = match.group(1)
                    current = None
                    if number in ('3.1', '3.2', '3.3', '4', '5') or number.startswith('5.'):
                        current = {'number': number, 'section': (number + ' ' + (match.group(2) or '')).strip(),
                                   'page': page, 'lines': [line]}
                        sections.append(current)
                elif current: current['lines'].append(line)
        for sec in sections:
            number = sec['number']
            content = '\n'.join(sec['lines']).strip()
            pid = _SECTION_IDS.get(number)
            if pid is None and number.startswith('3.'):
                match = _PATIENT_ID.search(content)
                if match: pid = match.group(1)
            docs.extend(_chunks(content, filename, doc_type, sec['page'], sec['section'], pid))
    if not docs:
        raise IngestError('Scanned PDF without text is not supported' if doc_type == 'uploaded_report'
                          else 'No extractable knowledge-base sections found')
    return docs


def find_patient(df, text: str) -> dict | None:
    if df is None or not isinstance(text, str) or not text.strip(): return None
    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        name = str(row['patient_name']).strip()
        if re.search(r'(?<!\w)' + re.escape(pid) + r'(?!\w)', text, re.I): return row.to_dict()
        # Whole-word matching avoids Doe inside Does; full names and first/last names supported.
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I): return row.to_dict()
        if any(re.search(r'(?<!\w)' + re.escape(part) + r'(?!\w)', text, re.I)
               for part in name.split() if part): return row.to_dict()
    return None


def list_patients(df) -> list[dict]:
    if df is None: return []
    return [{'patient_id': str(row['patient_id']), 'patient_name': str(row['patient_name'])}
            for _, row in df.iterrows()]


def load_any(file, filename: str) -> dict:
    ext = Path(filename).suffix.lower()
    if ext in ('.csv', '.xlsx', '.xls'):
        df = load_tabular(file, filename)
        return {'docs': df_to_documents(df, filename), 'df': df, 'kind': 'tabular'}
    if ext == '.pdf':
        kind = 'knowledge_base' if Path(filename).name.lower() == 'medical_reports_knowledge_base.pdf' else 'uploaded_report'
        return {'docs': load_pdf(file, filename, kind), 'df': None, 'kind': kind}
    raise IngestError('Unsupported file type. Use .csv, .xlsx, .xls or .pdf')
