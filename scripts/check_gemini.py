"""Verify your API key and model names.   Run:  python scripts/check_gemini.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
import config

if not config.credentials_configured():
    sys.exit("FAIL: " + config.credentials_hint())

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

candidates = [config.GEMINI_MODEL] + [m for m in ("gemini-2.5-flash", "gemini-2.5-flash-lite")
                                      if m != config.GEMINI_MODEL]
working = None
for m in candidates:
    try:
        out = ChatGoogleGenerativeAI(model=m, temperature=0).invoke("Reply with the single word: ready")
        print(f"  chat model OK   : {m} -> {str(out.content).strip()[:40]}")
        working = working or m
        if m == config.GEMINI_MODEL:
            break
    except Exception as e:  # noqa: BLE001
        print(f"  chat model FAIL : {m} -> {str(e)[:110]}")
if not working:
    sys.exit("FAIL: no chat model worked. Check your GOOGLE_API_KEY and quota.")
if working != config.GEMINI_MODEL:
    print(f"\n>>> Set GEMINI_MODEL={working} in your .env / Streamlit secrets.")

try:
    v = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL).embed_query("metformin 500mg")
    print(f"  embedding OK    : {config.EMBED_MODEL} -> vector length {len(v)}")
except Exception as e:  # noqa: BLE001
    sys.exit(f"FAIL embeddings ({config.EMBED_MODEL}): {str(e)[:200]}")
print("\nOK: API key and models work.")
