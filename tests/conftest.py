"""Shared fixtures: a fresh demo database in a temp folder and a fake document search (no model downloads)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.db_manager import DBManager  # noqa: E402
from backend.demo_seeder import build_database  # noqa: E402


class FakeRAG:
    """Stands in for ChromaDB so tests run offline and fast."""

    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [
            {"source": "employee_handbook.pdf", "page": 3, "score": 0.9,
             "text": "2.2 Full-Time Remote eligibility. An employee qualifies only if ALL criteria are met."},
        ]
        self.queries = []

    def search(self, query, k=4):
        self.queries.append(query)
        return self.hits[:k]

    def list_documents(self):
        return [{"name": "employee_handbook.pdf", "pages": 5, "chunks": 6}]


@pytest.fixture
def demo_db_path(tmp_path):
    path = tmp_path / "demo_hr.db"
    build_database(path)
    return path


@pytest.fixture
def db(demo_db_path):
    manager = DBManager(f"sqlite:///{demo_db_path}")
    yield manager
    manager.dispose()


@pytest.fixture
def rag():
    return FakeRAG()
