"""LLM layer: one LiteLLM-powered agent loop with SQL + PDF tools and 1-step SQL self-correction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import litellm
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from backend.db_manager import DBManager, UnsafeQueryError
from backend.rag_manager import RAGManager

litellm.drop_params = True          # ignore params a provider doesn't support
litellm.suppress_debug_info = True

MODELS = {
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4-5",
    "Gemini 3.8 Flash": "gemini/gemini-3.8-flash",
    "GPT-4o": "gpt-4o",
}
KEY_ENV = {
    "Claude Sonnet 4.5": "ANTHROPIC_API_KEY",
    "GPT-4o": "OPENAI_API_KEY",
    "Gemini 3.8 Flash": "GEMINI_API_KEY",
}
DEFAULT_MODEL = "Claude Sonnet 4.5"
MAX_STEPS = 6        # LLM turns before we force a final answer
ROWS_TO_LLM = 40     # rows sent back to the model (the UI shows all rows)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_sql_database",
            "description": (
                "Run ONE read-only SQL SELECT against the connected database and get the rows back. "
                "Use it for any facts, numbers, lists or aggregates stored in tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement using exact table/column names."},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pdf_documents",
            "description": (
                "Semantic search over the indexed PDF documents (e.g. the employee handbook). "
                "Use it for policies, rules, eligibility criteria, limits and definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in natural language."},
                    "top_k": {"type": "integer", "description": "Number of passages (1-8). Default 4."},
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class SQLRun:
    sql: str
    ok: bool
    df: pd.DataFrame | None = None
    error: str | None = None
    corrected_from: str | None = None   # original SQL if self-correction happened
    first_error: str | None = None


@dataclass
class AgentResult:
    answer: str
    model: str
    sql_runs: list[SQLRun] = field(default_factory=list)
    pdf_hits: list[dict] = field(default_factory=list)
    steps: int = 0


def _short_error(e: Exception) -> str:
    return str(getattr(e, "orig", None) or e).strip().splitlines()[0][:400]


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.S | re.I)
    return (m.group(1) if m else s).strip()


class QueryAgent:
    def __init__(self, model_label: str, db: DBManager | None, rag: RAGManager, api_key: str | None = None):
        self.label = model_label
        self.model = MODELS[model_label]
        self.db = db
        self.rag = rag
        self.api_key = api_key or None

    def _complete(self, messages: list[dict], **kwargs):
        # Gemini 3 models can loop (look "stuck") below temperature 1.0, so leave them on their default.
        if "gemini-3" not in self.model:
            kwargs.setdefault("temperature", 0)
        else:
            # Gemini 3 "thinks" at a high level by default, which can take over a minute per step.
            kwargs.setdefault("reasoning_effort", "low")
        # Fail with a clear error after 90s instead of hanging forever.
        return litellm.completion(model=self.model, messages=messages, api_key=self.api_key,
                                  timeout=90, num_retries=1, **kwargs)

    def _system_prompt(self) -> str:
        docs = ", ".join(d["name"] for d in self.rag.list_documents()) or "none"
        if self.db:
            db_part = f"SQL dialect: {self.db.dialect}\n\nDATABASE SCHEMA:\n{self.db.schema_text}"
        else:
            db_part = "DATABASE: not connected."
        return f"""You are QuickQuery AI, a precise analyst that answers questions using a SQL database and PDF documents.
Today's date: {date.today().isoformat()}

{db_part}

INDEXED PDF DOCUMENTS: {docs}

RULES
1. Data questions -> query_sql_database. Policy / document questions -> search_pdf_documents.
2. Hybrid questions (e.g. "who qualifies per our policy"): FIRST search the PDFs for the exact criteria,
   THEN write ONE SQL query that applies EVERY criterion, THEN answer. Never invent policy rules.
3. SQL: a single SELECT only, exact names from the schema, dialect-correct date functions,
   LIMIT 200 for row lists. Return readable columns (names, not only ids).
4. Final answer: short plain English. Lead with the direct answer and key numbers, list the policy
   criteria you applied, and cite documents as (file name, p. N). The UI already shows the result table
   and SQL, so do not paste the whole table or the SQL.
5. If the tools return nothing relevant, say so honestly."""

    # ---------- tools ----------
    def _run_sql(self, sql: str, result: AgentResult, on_step: Callable[[str], None]) -> str:
        if self.db is None:
            return json.dumps({"error": "No database is connected."})
        on_step("Running SQL query")
        try:
            run = SQLRun(sql=sql, ok=True, df=self.db.run_query(sql))
        except UnsafeQueryError as e:
            result.sql_runs.append(SQLRun(sql=sql, ok=False, error=str(e)))
            return json.dumps({"error": f"Blocked by read-only guardrail: {e}"})
        except SQLAlchemyError as e:
            err = _short_error(e)
            on_step("SQL error - self-correcting")
            fixed = sql
            try:
                fixed = self._fix_sql(sql, err)
                run = SQLRun(sql=fixed, ok=True, df=self.db.run_query(fixed), corrected_from=sql, first_error=err)
            except Exception as e2:  # one retry only
                err2 = _short_error(e2)
                result.sql_runs.append(SQLRun(sql=fixed, ok=False, error=err2, corrected_from=sql, first_error=err))
                return json.dumps({"error": f"Query failed: {err}. Automatic fix also failed: {err2}. "
                                            "Explain the problem to the user instead of retrying."})
        result.sql_runs.append(run)
        df = run.df
        payload = {
            "row_count": len(df),
            "columns": list(df.columns),
            "rows": df.head(ROWS_TO_LLM).to_dict(orient="records"),
        }
        if len(df) > ROWS_TO_LLM:
            payload["note"] = f"Only the first {ROWS_TO_LLM} of {len(df)} rows are shown to you."
        if run.corrected_from:
            payload["self_corrected_sql"] = run.sql
        return json.dumps(payload, default=str)

    def _fix_sql(self, sql: str, error: str) -> str:
        prompt = [
            {"role": "system", "content": (
                f"You fix broken {self.db.dialect} SQL. Reply with ONLY the corrected single SELECT "
                f"statement - no explanation, no markdown.\n\nSCHEMA:\n{self.db.schema_text}")},
            {"role": "user", "content": f"Query:\n{sql}\n\nDatabase error:\n{error}"},
        ]
        return _strip_fences(self._complete(prompt).choices[0].message.content)

    def _search_pdf(self, query: str, top_k: int, result: AgentResult, on_step: Callable[[str], None]) -> str:
        on_step(f'Searching documents: "{query}"')
        hits = self.rag.search(query, k=max(1, min(int(top_k or 4), 8)))
        seen = {(h["source"], h["page"], h["text"]) for h in result.pdf_hits}
        result.pdf_hits.extend(h for h in hits if (h["source"], h["page"], h["text"]) not in seen)
        if not hits:
            return json.dumps({"results": [], "note": "No documents indexed or nothing relevant found."})
        return json.dumps({"results": [{"source": h["source"], "page": h["page"], "text": h["text"]} for h in hits]})

    def _dispatch(self, name: str, raw_args: str, result: AgentResult, on_step) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "Tool arguments were not valid JSON."})
        if name == "query_sql_database":
            return self._run_sql(args.get("sql", ""), result, on_step)
        if name == "search_pdf_documents":
            return self._search_pdf(args.get("query", ""), args.get("top_k", 4), result, on_step)
        return json.dumps({"error": f"Unknown tool {name}"})

    # ---------- main loop ----------
    def run(self, question: str, history: list[dict] | None = None,
            on_step: Callable[[str], None] = lambda _s: None) -> AgentResult:
        messages = [{"role": "system", "content": self._system_prompt()}]
        for m in (history or [])[-6:]:
            if m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})
        result = AgentResult(answer="", model=self.label)

        for step in range(1, MAX_STEPS + 1):
            result.steps = step
            on_step(f"Thinking with {self.label} (step {step})")
            msg = self._complete(messages, tools=TOOLS, tool_choice="auto").choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                result.answer = (msg.content or "").strip() or "I could not produce an answer."
                return result
            # Append LiteLLM's own message object so provider extras survive
            # (Gemini 3 rejects the next call if its "thought signatures" are dropped).
            messages.append(msg)
            for c in calls:
                output = self._dispatch(c.function.name, c.function.arguments, result, on_step)
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": output})

        # Step budget used up: ask for a final answer without more tool calls.
        messages.append({"role": "user", "content": "Stop using tools and answer now with what you have."})
        msg = self._complete(messages, tools=TOOLS, tool_choice="none").choices[0].message
        result.answer = (msg.content or "").strip() or "I could not finish within the step limit."
        return result
