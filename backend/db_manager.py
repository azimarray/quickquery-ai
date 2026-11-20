"""SQL layer: multi-database connections, schema inspection and a read-only guardrail."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

DB_TYPES = {
    "SQLite": "sqlite",
    "MySQL": "mysql+mysqlconnector",
    "PostgreSQL": "postgresql+psycopg2",
}
DEFAULT_PORTS = {"MySQL": 3306, "PostgreSQL": 5432}
MAX_TABLES = 15          # tables described to the LLM
MAX_ROWS = 1000          # hard cap on rows fetched per query
MAX_ENUM_VALUES = 12     # list distinct values for low-cardinality text columns

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|grant|revoke|attach|detach|"
    r"pragma|vacuum|reindex|exec|execute|call|copy|lock|into|outfile|load_file)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


def validate_select(sql: str) -> str:
    """Return a cleaned SQL string, or raise UnsafeQueryError if it is not read-only."""
    cleaned = re.sub(r"/\*.*?\*/", " ", sql or "", flags=re.S)
    cleaned = re.sub(r"--[^\n]*", " ", cleaned).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Empty query.")
    # Ignore keywords that only appear inside string literals / quoted identifiers.
    bare = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    bare = re.sub(r'"(?:[^"]|"")*"', '""', bare)
    bare = re.sub(r"`[^`]*`", "``", bare)
    if ";" in bare:
        raise UnsafeQueryError("Only one statement is allowed.")
    if not re.match(r"^\s*\(?\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    hit = FORBIDDEN.search(bare)
    if hit:
        raise UnsafeQueryError(f"Keyword '{hit.group(0).upper()}' is not allowed (read-only mode).")
    return cleaned


def build_url(db_type: str, host: str = "", port: str | int | None = None, user: str = "",
              password: str = "", database: str = "", sqlite_path: Path | str | None = None) -> URL:
    if db_type == "SQLite":
        return URL.create("sqlite", database=str(sqlite_path))
    return URL.create(
        DB_TYPES[db_type],
        username=user or None,
        password=password or None,
        host=host or "localhost",
        port=int(port) if port else DEFAULT_PORTS[db_type],
        database=database or None,
    )


def _is_text(type_name: str) -> bool:
    return any(k in type_name.upper() for k in ("CHAR", "TEXT", "STRING", "ENUM"))


class DBManager:
    def __init__(self, url: str | URL):
        url = make_url(url)
        backend = url.get_backend_name()
        connect_args = {
            "sqlite": {"check_same_thread": False},
            "mysql": {"connection_timeout": 5},
            "postgresql": {"connect_timeout": 5},
        }.get(backend, {})
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        if backend == "sqlite":
            @event.listens_for(self.engine, "connect")
            def _read_only(dbapi_conn, _record):  # second safety net for SQLite
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA query_only = ON")
                cur.close()

        self.dialect = self.engine.dialect.name
        self.display_url = url.render_as_string(hide_password=True)
        self.schema = self._inspect()          # {table: [{name, type, pk}]}
        self.schema_text = self._schema_text()  # compact description for the LLM

    # ---------- schema ----------
    def _inspect(self) -> dict[str, list[dict]]:
        insp = inspect(self.engine)
        schema = {}
        for table in insp.get_table_names()[:MAX_TABLES]:
            pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            cols = []
            for c in insp.get_columns(table):
                try:
                    type_name = str(c["type"])
                except Exception:
                    type_name = "UNKNOWN"
                cols.append({"name": c["name"], "type": type_name, "pk": c["name"] in pk})
            schema[table] = cols
        return schema

    def _schema_text(self) -> str:
        quote = self.engine.dialect.identifier_preparer.quote
        blocks = []
        with self.engine.connect() as conn:
            for table, cols in self.schema.items():
                parts = []
                for c in cols:
                    part = f"{c['name']} {c['type']}{' PRIMARY KEY' if c['pk'] else ''}"
                    if _is_text(c["type"]):
                        try:
                            sql = (f"SELECT DISTINCT {quote(c['name'])} FROM {quote(table)} "
                                   f"LIMIT {MAX_ENUM_VALUES + 1}")
                            values = [r[0] for r in conn.execute(text(sql)) if r[0] is not None]
                            if 0 < len(values) <= MAX_ENUM_VALUES:
                                part += f"  -- values: {', '.join(map(str, values))}"
                        except SQLAlchemyError:
                            conn.rollback()
                    parts.append(part)
                blocks.append(f"TABLE {table} (\n  " + ",\n  ".join(parts) + "\n)")
        return "\n\n".join(blocks) or "(no tables found)"

    # ---------- queries ----------
    def run_query(self, sql: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
        safe_sql = validate_select(sql)
        with self.engine.connect() as conn:
            if self.dialect == "postgresql":
                conn = conn.execution_options(postgresql_readonly=True)
            result = conn.execute(text(safe_sql))
            rows = result.fetchmany(max_rows)
            df = pd.DataFrame(rows, columns=list(result.keys()))
            conn.rollback()
        return df

    def dispose(self) -> None:
        self.engine.dispose()
