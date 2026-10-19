# 🩺 Medical Report Assistant

An AI-powered application that helps users understand medical information through natural-language questions.

Built using **Streamlit, LangChain, Gemini 2.5 Flash, and FAISS**, the project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-based answers.

## ✨ Features

- 💬 Ask questions about medical information.
- 📄 Process supported medical documents and patient data.
- 🔎 Retrieve relevant information using FAISS.
- 🤖 Generate answers using Gemini 2.5 Flash.
- 📊 Support summarization and analytics.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini 2.5 Flash
- FAISS
- Pytest

## 📁 Project Structure

```text
├── .devcontainer/
├── .streamlit/
├── Test_sample data sets/
├── data/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── app.py
├── analytics.py
├── config.py
├── conftest.py
├── ingest.py
├── mocks.py
├── rag.py
├── requirements.txt
└── summarize.py
```

## 🚀 Run Locally

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

**2. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Copy `.env.example` to `.env` and add your Gemini API key.

```env
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

**5. Run the application**

```bash
streamlit run app.py
```

## 🧪 Run Tests

```bash
pytest tests -v
```

## ☁️ Deployment

Deploy using [Streamlit Community Cloud](https://share.streamlit.io/). Add your required API key and model configuration under the app's Secrets settings.

## 🔐 Disclaimer

This project supports medical information understanding and is not a substitute for professional medical advice, diagnosis, or treatment.

- **Live Demo:** https://medical-agent-agenticai.streamlit.app
