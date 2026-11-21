"""The database must stay read-only, whatever SQL the AI writes."""
import pytest

from backend.db_manager import UnsafeQueryError, validate_select


@pytest.mark.parametrize("sql", [
    "SELECT * FROM employees",
    "select count(*) from employees;",
    "WITH t AS (SELECT 1 AS x) SELECT * FROM t",
    "SELECT REPLACE(full_name, 'a', 'b') FROM employees",
    "SELECT * FROM employees WHERE job_title = 'Please delete me'",
    "SELECT 1 -- ; DROP TABLE employees",
])
def test_allows_read_only_queries(sql):
    assert validate_select(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM employees",
    "UPDATE employees SET salary = 0",
    "DROP TABLE employees",
    "INSERT INTO employees (full_name) VALUES ('x')",
    "SELECT 1; DROP TABLE employees",
    "SELECT * INTO backup FROM employees",
    "PRAGMA table_info(employees)",
    "WITH x AS (DELETE FROM employees RETURNING *) SELECT * FROM x",
    "ATTACH DATABASE 'other.db' AS other",
    "",
])
def test_blocks_anything_that_could_change_data(sql):
    with pytest.raises(UnsafeQueryError):
        validate_select(sql)


def test_blocked_query_never_reaches_the_database(db):
    before = db.run_query("SELECT COUNT(*) AS n FROM employees")["n"][0]
    with pytest.raises(UnsafeQueryError):
        db.run_query("DELETE FROM employees")
    assert db.run_query("SELECT COUNT(*) AS n FROM employees")["n"][0] == before


def test_sqlite_connection_itself_is_read_only(db):
    """Second safety net: even if the guard were bypassed, SQLite refuses writes."""
    with db.engine.connect() as conn:
        with pytest.raises(Exception):
            conn.exec_driver_sql("DELETE FROM employees")


def test_row_cap(db):
    df = db.run_query("SELECT * FROM employees", max_rows=5)
    assert len(df) == 5
