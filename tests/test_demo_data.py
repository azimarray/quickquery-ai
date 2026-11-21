"""Demo data is generated on first run, so it has to be correct and repeatable."""
from datetime import date

from pypdf import PdfReader

from backend.demo_seeder import build_handbook, employee_rows
from backend.rag_manager import chunk_text


def test_employee_rows_are_repeatable_and_unique():
    today = date(2026, 1, 1)
    first, second = employee_rows(today), employee_rows(today)
    assert first == second
    assert len(first) == 30
    assert len({r["full_name"] for r in first}) == 30
    assert len({r["email"] for r in first}) == 30


def test_employee_values_are_valid():
    for r in employee_rows(date(2026, 1, 1)):
        assert r["employment_type"] in {"Full-Time", "Part-Time", "Contractor"}
        assert 1 <= r["performance_rating"] <= 5
        assert r["pto_days_used"] + r["pto_days_remaining"] == {"Full-Time": 25, "Part-Time": 12,
                                                                "Contractor": 0}[r["employment_type"]]


def test_schema_is_inspected_for_the_ai(db):
    assert "employees" in db.schema
    names = {c["name"] for c in db.schema["employees"]}
    assert {"full_name", "department", "hire_date", "pto_days_remaining"} <= names
    # low-cardinality text columns list their values so the AI uses exact spellings
    assert "Customer Support" in db.schema_text
    assert "Full-Time" in db.schema_text


def test_handbook_pdf_has_the_policies(tmp_path):
    path = tmp_path / "employee_handbook.pdf"
    build_handbook(path)
    pages = [" ".join(p.extract_text().split()) for p in PdfReader(str(path)).pages]
    assert len(pages) == 5
    assert "Remote Work Policy" in pages[2]
    assert "maximum of 5 unused PTO days" in pages[3]


def test_chunking_keeps_all_text_in_bounded_chunks():
    text = "This is a sentence about policy. " * 200
    chunks = chunk_text(text, size=900, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)
    assert chunks[-1].endswith("policy.")
