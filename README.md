# Medical Report Assistant

Streamlit + LangChain + Gemini 2.5 Flash + FAISS.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then paste your own GOOGLE_API_KEY into .env
python scripts/check_gemini.py                           # verifies key + model names
pytest tests -v                                          # offline tests
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub (`.env` is git-ignored, so your key is not uploaded).
2. On https://share.streamlit.io choose the repo and `app.py` as the main file.
3. In **App settings -> Secrets** add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```

## Layout
| File | Purpose |
|---|---|
| app.py | Streamlit UI |
| ingest.py | Load patient data / PDFs |
| rag.py | Embeddings, FAISS index, question answering |
| summarize.py, analytics.py | Summaries, trends, safety rules |
| config.py | Model names and settings |
| scripts/ | Key check, sample data, smoke test |
| tests/ | Unit tests |

Never commit `.env`.
