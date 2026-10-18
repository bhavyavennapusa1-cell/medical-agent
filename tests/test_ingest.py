import io
import pandas as pd
import pytest
from reportlab.pdfgen import canvas
from ingest import (IngestError, load_any, load_tabular, load_pdf, find_patient,
                    list_patients, df_to_documents)

@pytest.fixture
def df():
    return pd.DataFrame([
        {'patient_id':'101','patient_name':'John Doe','age':45,'diagnosis':'Hypertension','notes':'BP 140/90'},
        {'patient_id':'102','patient_name':'Jane Smith','age':34,'diagnosis':'Diabetes','notes':'HbA1c 7.5%'},
        {'patient_id':'103','patient_name':'Robert Johnson','age':52,'diagnosis':'Asthma','notes':'Wheezing'},
    ])

def pdf_bytes(lines):
    buf=io.BytesIO(); c=canvas.Canvas(buf)
    for page_lines in lines:
        y=760
        for line in page_lines: c.drawString(40,y,line);y-=22
        c.showPage()
    c.save();return buf.getvalue()

def test_csv_and_contract(df):
    data=df.to_csv(index=False).encode()
    result=load_any(data,'patients.csv')
    assert len(result['docs'])==3 and len(result['df'])==3
    assert result['kind']=='tabular'
    assert set(result['docs'][0].metadata)=={'source','doc_type','page','section','patient_id','patient_name'}
    assert result['docs'][1].metadata['patient_id']=='102'

def test_excel_first_nonempty_sheet(df):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        pd.DataFrame().to_excel(writer,sheet_name='Sheet1',index=False)
        df.to_excel(writer,sheet_name='1',index=False)
    result=load_tabular(buf.getvalue(),'medical_reports_csv.xlsx')
    assert len(result)==3 and result.loc[1,'patient_id']=='102'

def test_find_patient_whole_words(df):
    assert find_patient(df,'jane')['patient_id']=='102'
    assert find_patient(df,'SMITH')['patient_id']=='102'
    assert find_patient(df,'102')['patient_id']=='102'
    assert find_patient(df,'Does') is None
    assert find_patient(df,'John Doe')['patient_id']=='101'
    assert len(list_patients(df))==3

def test_kb_skips_table_and_tags_sections():
    data=pdf_bytes([['Patient table: Recomme (truncated)','3.1 Hypertension (Patient ID 101)',
                     'Blood pressure 140/90','3.2 Diabetes (Patient ID 102)',
                     'HbA1c 7.5%','3.3 Asthma (Patient ID 103)','Wheezing',
                     '4. Medication safety','Check contraindications',
                     'Section 5 Retrieval metadata','Source references']])
    docs=load_pdf(data,'medical_reports_knowledge_base.pdf','knowledge_base')
    assert {'101','102','103'} <= {d.metadata['patient_id'] for d in docs}
    assert all('Recomme' not in d.page_content for d in docs)
    assert any(d.metadata['patient_id'] is None for d in docs)

def test_uploaded_pdf_and_empty_pdf():
    docs=load_pdf(pdf_bytes([['Glucose 142 mg/dL']]),'report.pdf','uploaded_report')
    assert docs[0].metadata['page']==1
    with pytest.raises(IngestError,match='Scanned PDF'):
        load_pdf(pdf_bytes([[]]),'blank.pdf','uploaded_report')

def test_invalid_and_empty():
    with pytest.raises(IngestError):load_any(b'','patients.csv')
    with pytest.raises(IngestError):load_any(b'x','malware.exe')
    with pytest.raises(IngestError,match='Missing required columns'):
        load_tabular(b'patient_id,age\n101,20\n','bad.csv')
