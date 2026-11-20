# ⚡ QuickQuery AI
**Hybrid Text-to-SQL & Document Intelligence Engine** — ask one question, get an answer that combines your SQL database and your PDFs.

Stack: Streamlit · LiteLLM (Claude Sonnet 4.5 / GPT-4o / Gemini 3.8 Flash) · SQLAlchemy (SQLite / MySQL / PostgreSQL) · ChromaDB · pypdf.

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
cp .env.example .env                                 # add at least ANTHROPIC_API_KEY
streamlit run app.py
```
First launch creates the demo HR database and handbook PDF, and downloads ChromaDB's small embedding model (~80 MB, once).

## Deploy to Streamlit Community Cloud
```bash
git init && git add . && git commit -m "QuickQuery AI"
git branch -M main
git remote add origin https://github.com/<you>/quickquery-ai.git
git push -u origin main
```
Then on https://share.streamlit.io → **Create app** → pick the repo, main file `app.py` → **Advanced settings → Secrets**:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
```
Demo data is regenerated automatically whenever the cloud container restarts.

## Safety
Every query passes a guardrail (single statement, must start with SELECT/WITH, write keywords blocked), and runs in a read-only session (SQLite `query_only`, PostgreSQL read-only transaction, always rolled back).
