"""The AI agent loop, tested with a scripted fake LLM so no API key or network is needed."""
import json
from types import SimpleNamespace

import pytest

import backend.llm_router as router
from backend.llm_router import AgentResult, QueryAgent, SQLRun, memory_text, run_with_fallback


def tool_call(name, args, call_id="call_1"):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


@pytest.fixture
def fake_llm(monkeypatch):
    """Queue up LLM replies; record every call's arguments."""
    state = {"replies": [], "calls": []}

    def completion(**kwargs):
        state["calls"].append(kwargs)
        return state["replies"].pop(0)

    monkeypatch.setattr(router.litellm, "completion", completion)
    return state


def test_hybrid_question_uses_pdf_then_sql(db, rag, fake_llm):
    fake_llm["replies"] = [
        reply(calls=[tool_call("search_pdf_documents", {"query": "remote work eligibility"})]),
        reply(calls=[tool_call("query_sql_database",
                               {"sql": "SELECT full_name FROM employees WHERE performance_rating >= 4"}, "call_2")]),
        reply("Ten employees qualify (employee_handbook.pdf, p. 3)."),
    ]
    result = QueryAgent("Claude Sonnet 4.5 (Anthropic)", db, rag).run("Who qualifies for remote work?")
    assert result.answer.startswith("Ten employees qualify")
    assert result.steps == 3
    assert rag.queries == ["remote work eligibility"]
    assert len(result.sql_runs) == 1 and result.sql_runs[0].ok and len(result.sql_runs[0].df) > 0
    assert result.pdf_hits[0]["page"] == 3
    # the SQL rows were sent back to the model as a tool result
    tool_msgs = [m for m in fake_llm["calls"][-1]["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("row_count" in m["content"] for m in tool_msgs)


def test_broken_sql_is_fixed_once_automatically(db, rag, fake_llm):
    fake_llm["replies"] = [
        reply(calls=[tool_call("query_sql_database", {"sql": "SELECT nme FROM employees"})]),
        reply("```sql\nSELECT full_name FROM employees\n```"),  # the 1-step fix
        reply("Here are the employees."),
    ]
    result = QueryAgent("Claude Sonnet 4.5 (Anthropic)", db, rag).run("List employees")
    run = result.sql_runs[0]
    assert run.ok and run.corrected_from == "SELECT nme FROM employees"
    assert run.sql == "SELECT full_name FROM employees"
    assert "nme" in run.first_error
    assert len(run.df) == 30


def test_ai_cannot_change_data(db, rag, fake_llm):
    fake_llm["replies"] = [
        reply(calls=[tool_call("query_sql_database", {"sql": "DELETE FROM employees"})]),
        reply("I can't do that, the database is read-only."),
    ]
    result = QueryAgent("Claude Sonnet 4.5 (Anthropic)", db, rag).run("Delete everyone")
    assert not result.sql_runs[0].ok
    assert db.run_query("SELECT COUNT(*) AS n FROM employees")["n"][0] == 30


def test_reasoning_models_use_low_effort_others_temperature_zero(db, rag, fake_llm):
    fake_llm["replies"] = [reply("ok"), reply("ok")]
    QueryAgent("GPT-OSS 120B (Groq)", db, rag).run("hi")
    QueryAgent("Claude Sonnet 4.5 (Anthropic)", db, rag).run("hi")
    groq, claude = fake_llm["calls"]
    assert groq["reasoning_effort"] == "low" and "temperature" not in groq
    assert claude["temperature"] == 0 and "reasoning_effort" not in claude
    assert groq["timeout"] == 90


def test_backup_model_answers_when_the_first_fails(db, rag, monkeypatch):
    tried = []

    def fake_run(self, question, history=None, on_step=lambda s: None):
        tried.append(self.label)
        if self.label == "GPT-OSS 120B (Groq)":
            raise RuntimeError("rate limited")
        return AgentResult(answer="ok", model=self.label)

    monkeypatch.setattr(QueryAgent, "run", fake_run)
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    result = run_with_fallback("GPT-OSS 120B (Groq)", db, rag, "q", custom_keys={"GPT-OSS 120B (Groq)": "k"})
    assert tried == ["GPT-OSS 120B (Groq)", "Gemini 3.8 Flash (Google)"]  # Claude skipped: no key
    assert result.model == "Gemini 3.8 Flash (Google)"
    assert result.fallback_from == "GPT-OSS 120B (Groq)"


def test_error_names_the_model_when_every_option_fails(db, rag, monkeypatch):
    def always_fail(self, *a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(QueryAgent, "run", always_fail)
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError) as err:
        run_with_fallback("GPT-OSS 120B (Groq)", db, rag, "q")
    assert err.value.qq_model == "GPT-OSS 120B (Groq)"


def test_memory_carries_past_sql_and_pdf_context(db, rag, fake_llm):
    past = AgentResult(answer="10 employees qualify.", model="x",
                       sql_runs=[SQLRun(sql="SELECT full_name FROM employees WHERE performance_rating >= 4", ok=True)],
                       pdf_hits=rag.hits)
    text = memory_text({"role": "assistant", "content": past.answer, "result": past})
    assert "[SQL used]" in text and "performance_rating >= 4" in text
    assert "employee_handbook.pdf p.3" in text

    history = [{"role": "user", "content": f"question {i}"} if i % 2 == 0 else
               {"role": "assistant", "content": f"answer {i}"} for i in range(30)]
    fake_llm["replies"] = [reply("ok")]
    QueryAgent("Claude Sonnet 4.5 (Anthropic)", db, rag).run("follow-up", history=history)
    sent = fake_llm["calls"][0]["messages"]
    assert len(sent) == 1 + 20 + 1  # system prompt + last 20 messages + new question
    assert sent[1]["content"] == "question 10" and sent[-1]["content"] == "follow-up"
