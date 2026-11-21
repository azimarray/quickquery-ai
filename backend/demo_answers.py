"""Instant answers for the 3 starter prompts on the built-in demo data.

They run real SQL and a real PDF search, but no LLM call, so a live demo can never stall on an
overloaded or rate-limited model. Typed questions always go to the live AI.
"""
from __future__ import annotations

from backend.db_manager import DBManager
from backend.llm_router import AgentResult, SQLRun
from backend.rag_manager import RAGManager

HYBRID = ("Which employees qualify for full-time remote work per our handbook policy, "
          "and how many PTO days do they have left?")
SQL_ONLY = "What is the average salary and headcount per department?"
PDF_ONLY = "What is the maximum number of PTO days that can carry over to next year?"
STARTERS = [("Hybrid · SQL + PDF", HYBRID), ("SQL only", SQL_ONLY), ("PDF only", PDF_ONLY)]

LABEL = "Instant demo answer (pre-built SQL + PDF search, no LLM call)"

REMOTE_SQL = """SELECT full_name, department, job_title, hire_date, performance_rating, pto_days_remaining
FROM employees
WHERE employment_type = 'Full-Time'
  AND hire_date <= date('now', '-2 years')
  AND performance_rating >= 4
  AND department NOT IN ('Operations', 'Customer Support')
ORDER BY pto_days_remaining DESC"""

DEPT_SQL = """SELECT department, COUNT(*) AS headcount, ROUND(AVG(salary), 0) AS avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC"""


def _page(hits: list[dict], fallback: int) -> str:
    return f"{hits[0]['source']}, p. {hits[0]['page']}" if hits else f"employee handbook, p. {fallback}"


def instant_answer(question: str, db: DBManager | None, rag: RAGManager) -> AgentResult | None:
    """Return a ready answer for a starter prompt on the demo SQLite DB, else None (use the live AI)."""
    if db is None or db.dialect != "sqlite" or not db.display_url.endswith("demo_hr.db"):
        return None
    result = AgentResult(answer="", model=LABEL, steps=0)

    if question == HYBRID:
        hits = rag.search("full-time remote work eligibility criteria", k=2)
        df = db.run_query(REMOTE_SQL)
        result.pdf_hits, result.sql_runs = hits, [SQLRun(sql=REMOTE_SQL, ok=True, df=df)]
        if df.empty:
            result.answer = "No employees currently meet all the full-time remote work criteria."
        else:
            pto = df["pto_days_remaining"]
            result.answer = (
                f"**{len(df)} employees qualify** for full-time remote work. Per the handbook "
                f"({_page(hits, 3)}), an employee must be **Full-Time**, have **at least 2 years** of service, "
                f"a performance rating of **4 or higher**, and **not** work in Operations or Customer Support.\n\n"
                f"Their remaining PTO ranges from **{pto.min()} to {pto.max()} days** "
                f"({int(pto.sum())} days in total). {df.iloc[0]['full_name']} has the most left "
                f"({pto.max()} days)."
            )
        return result

    if question == SQL_ONLY:
        df = db.run_query(DEPT_SQL)
        result.sql_runs = [SQLRun(sql=DEPT_SQL, ok=True, df=df)]
        top, low = df.iloc[0], df.iloc[-1]
        result.answer = (
            f"The company has **{int(df['headcount'].sum())} employees across {len(df)} departments**. "
            f"**{top['department']}** has the highest average salary (${int(top['avg_salary']):,}) and "
            f"**{low['department']}** the lowest (${int(low['avg_salary']):,}). "
            f"Team sizes range from {int(df['headcount'].min())} to {int(df['headcount'].max())} people."
        )
        return result

    if question == PDF_ONLY:
        hits = rag.search("maximum PTO days carry over to next year", k=2)
        result.pdf_hits = hits
        result.answer = (
            f"A maximum of **5 unused PTO days** can carry over to the next calendar year. Carried-over "
            f"days must be used by **March 31**, otherwise they are forfeited, and any unused days above "
            f"the 5-day limit are forfeited on December 31 ({_page(hits, 4)})."
        )
        return result

    return None
