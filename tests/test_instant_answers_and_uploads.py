"""Starter prompts must answer correctly without an AI call; uploaded spreadsheets become queryable tables."""
import io
import sqlite3
from datetime import date, timedelta

import pandas as pd
import pytest

from backend.db_manager import DBManager, drop_table, import_spreadsheet
from backend.demo_answers import HYBRID, PDF_ONLY, SQL_ONLY, instant_answer
from backend.demo_seeder import employee_rows


def test_hybrid_starter_matches_the_handbook_rules(db, rag):
    result = instant_answer(HYBRID, db, rag)
    today = date.today()
    try:  # same rule as SQLite's date('now', '-2 years')
        two_years_ago = today.replace(year=today.year - 2)
    except ValueError:  # 29 February
        two_years_ago = today.replace(year=today.year - 2, day=28) + timedelta(days=1)
    expected = {
        r["full_name"] for r in employee_rows()
        if r["employment_type"] == "Full-Time" and r["hire_date"] <= two_years_ago
        and r["performance_rating"] >= 4 and r["department"] not in ("Operations", "Customer Support")
    }
    df = result.sql_runs[0].df
    assert set(df["full_name"]) == expected
    assert f"{len(expected)} employees qualify" in result.answer
    assert result.steps == 0 and "no LLM call" in result.model
    assert result.pdf_hits, "the answer should cite the handbook"


def test_sql_starter_totals_add_up(db, rag):
    result = instant_answer(SQL_ONLY, db, rag)
    df = result.sql_runs[0].df
    assert df["headcount"].sum() == 30
    assert "30 employees" in result.answer


def test_pdf_starter_cites_the_limit(db, rag):
    result = instant_answer(PDF_ONLY, db, rag)
    assert "5 unused PTO days" in result.answer
    assert not result.sql_runs


def test_typed_questions_go_to_the_live_ai(db, rag):
    assert instant_answer("How many engineers are there?", db, rag) is None
    assert instant_answer(HYBRID, None, rag) is None


def test_csv_upload_becomes_a_clean_table(demo_db_path):
    csv = b"Product Name,Units Sold,2026 Revenue ($)\nWidget,10,99.5\nGadget,5,40\n,,\n"
    assert import_spreadsheet(csv, "Q3 Sales Report.csv", demo_db_path) == [("q3_sales_report", 2)]
    db = DBManager(f"sqlite:///{demo_db_path}")
    assert {c["name"] for c in db.schema["q3_sales_report"]} == {"product_name", "units_sold", "col_3_2026_revenue"}
    assert db.run_query("SELECT SUM(units_sold) AS total FROM q3_sales_report")["total"][0] == 15
    db.dispose()


def test_excel_upload_makes_one_table_per_sheet_and_protects_demo_data(demo_db_path):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        pd.DataFrame({"Name": ["A", "B"]}).to_excel(writer, sheet_name="Team A", index=False)
        pd.DataFrame({"Name": ["C"]}).to_excel(writer, sheet_name="Team B", index=False)
    tables = import_spreadsheet(buf.getvalue(), "employees.xlsx", demo_db_path)
    assert tables == [("employees_team_a", 2), ("employees_team_b", 1)]
    single = import_spreadsheet(b"x\n1\n", "employees.csv", demo_db_path)
    assert single == [("upload_employees", 1)], "an upload must never replace the demo table"
    with sqlite3.connect(demo_db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 30


def test_removing_an_upload(demo_db_path):
    import_spreadsheet(b"a\n1\n", "extra.csv", demo_db_path)
    drop_table(demo_db_path, "extra")
    with sqlite3.connect(demo_db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "extra" not in names
    with pytest.raises(ValueError):
        drop_table(demo_db_path, "employees")
