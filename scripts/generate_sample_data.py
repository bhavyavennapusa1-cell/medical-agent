"""M4 sample data (Lead-corrected). Run from the repo root:  python scripts/generate_sample_data.py
Writes data/sample_lab_report.pdf (fictional patient 201: 5 rising glucose readings, HbA1c, BP, lipids, meds)
and data/sample_trend.csv (same readings; for get_trend tests - the app's CSV uploader needs the 5 patient columns)."""
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DATA = Path(__file__).resolve().parent.parent / "data"
DATES = ["2026-09-01", "2026-09-05", "2026-09-12", "2026-09-19", "2026-09-26"]
GLUCOSE = [110, 134, 158, 185, 210]


def main():
    DATA.mkdir(exist_ok=True)
    pd.DataFrame({"date": DATES, "test": "Glucose", "value": GLUCOSE, "unit": "mg/dL"}).to_csv(DATA / "sample_trend.csv", index=False)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=6, leading=14)
    story = [Paragraph("<b>LABORATORY AND CLINICAL REPORT (SAMPLE, FICTIONAL PATIENT)</b>", styles["Heading1"]),
             Paragraph("Patient Name: Terrance Vance | Patient ID: 201 | Age: 61", body),
             Paragraph("Primary Diagnosis: Type 2 Diabetes with metabolic syndrome", body), Spacer(1, 8),
             Paragraph("<b>Glucose readings</b>", body)]
    story += [Paragraph(f"Glucose | {d} | {v} mg/dL", body) for d, v in zip(DATES, GLUCOSE)]
    story += [Spacer(1, 8), Paragraph("<b>Current examinations</b>", body),
              Paragraph("HbA1c: 7.9 %", body),
              Paragraph("BP: 145/95 mmHg", body),
              Paragraph("Heart Rate: 82 bpm", body), Spacer(1, 8),
              Paragraph("<b>Lipid panel</b>", body),
              Paragraph("Total Cholesterol: 245 mg/dL", body),
              Paragraph("LDL: 162 mg/dL", body),
              Paragraph("HDL: 38 mg/dL", body), Spacer(1, 8),
              Paragraph("<b>Medications</b>", body),
              Paragraph("Metformin 1000mg orally once daily with dinner.", body), Spacer(1, 8),
              Paragraph("<b>Clinical notes</b>", body),
              Paragraph("Mild renal impairment noted with eGFR 44. Daily glucose self-monitoring advised. Follow-up in 4 weeks.", body)]
    SimpleDocTemplate(str(DATA / "sample_lab_report.pdf"), pagesize=letter).build(story)
    print("Wrote", DATA / "sample_lab_report.pdf", "and", DATA / "sample_trend.csv")


if __name__ == "__main__":
    main()
