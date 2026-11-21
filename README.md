# ⚡ QuickQuery AI
[![tests](https://github.com/azimarray/quickquery-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/azimarray/quickquery-ai/actions/workflows/tests.yml)

**Hybrid Text-to-SQL & Document Intelligence Engine** — ask one question, get an answer that combines your SQL database and your PDFs.

Stack: Streamlit · LiteLLM (GPT-OSS 120B on Groq - free, default / Claude Sonnet 4.5 (Anthropic) / Gemini 3.8 Flash (Google)) · SQLAlchemy (SQLite / MySQL / PostgreSQL) · ChromaDB · pypdf.

```
app.py                    Streamlit UI (sidebar controls + chat workspace)
backend/llm_router.py     Agent loop: tool calling, multi-step, 1-step SQL self-correction
backend/db_manager.py     DB connections, schema inspection, read-only SELECT guardrail
backend/rag_manager.py    PDF extraction, page-aware chunking, semantic search
backend/demo_seeder.py    Creates data/demo_hr.db + data/employee_handbook.pdf on first run
.streamlit/config.toml    Dark theme
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add at least GROQ_API_KEY (free at console.groq.com)
streamlit run app.py
```
First launch creates the demo HR database and handbook PDF, and downloads ChromaDB's small embedding model (~80 MB, once).

## Tests
```bash
pip install pytest
pytest -v
```
35+ automated tests run on every push (GitHub Actions) with a scripted fake LLM, so no API key is needed:
read-only SQL guard, demo data, instant starter answers, spreadsheet uploads, the agent loop (PDF + SQL tool calls),
1-step SQL self-correction, backup-model fallback and 20-message memory.

## Deploy to Streamlit Community Cloud
```bash
git init && git add . && git commit -m "QuickQuery AI"
git branch -M main
git remote add origin https://github.com/<you>/quickquery-ai.git
git push -u origin main
```
Then on https://share.streamlit.io → **Create app** → pick the repo, main file `app.py` → **Advanced settings → Secrets**:
```toml
GROQ_API_KEY = "gsk_..."
ANTHROPIC_API_KEY = "sk-ant-..."
GEMINI_API_KEY = "..."
```
Demo data is regenerated automatically whenever the cloud container restarts.

## Safety
Every query passes a guardrail (single statement, must start with SELECT/WITH, write keywords blocked), and runs in a read-only session (SQLite `query_only`, PostgreSQL read-only transaction, always rolled back).
