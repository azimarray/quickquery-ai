"""Creates the zero-setup demo data on first run: an HR SQLite database and an employee handbook PDF."""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF
from sqlalchemy import Column, Date, Integer, MetaData, String, Table, create_engine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "demo_hr.db"
PDF_PATH = DATA_DIR / "employee_handbook.pdf"

metadata = MetaData()
employees = Table(
    "employees", metadata,
    Column("employee_id", Integer, primary_key=True),
    Column("full_name", String(80), nullable=False),
    Column("email", String(120)),
    Column("department", String(40)),
    Column("job_title", String(60)),
    Column("employment_type", String(20)),   # Full-Time | Part-Time | Contractor
    Column("hire_date", Date),
    Column("salary", Integer),               # annual, USD
    Column("performance_rating", Integer),   # 1-5, most recent review
    Column("office_location", String(40)),
    Column("pto_days_used", Integer),        # current calendar year
    Column("pto_days_remaining", Integer),   # current calendar year
)

DEPARTMENTS = {
    "Engineering": (["Software Engineer", "Senior Software Engineer", "Engineering Manager", "DevOps Engineer"], 90_000, 170_000),
    "Data": (["Data Analyst", "Data Scientist", "Analytics Engineer"], 80_000, 150_000),
    "Sales": (["Account Executive", "Sales Manager", "Sales Development Rep"], 60_000, 130_000),
    "Marketing": (["Marketing Specialist", "Content Strategist", "Growth Manager"], 60_000, 120_000),
    "Finance": (["Financial Analyst", "Accountant", "Finance Manager"], 70_000, 135_000),
    "Human Resources": (["HR Generalist", "Recruiter", "HR Manager"], 55_000, 110_000),
    "Operations": (["Operations Coordinator", "Facilities Manager", "Logistics Specialist"], 50_000, 100_000),
    "Customer Support": (["Support Specialist", "Support Team Lead"], 45_000, 85_000),
}
FIRST = ["Aisha", "Liam", "Sofia", "Omar", "Mei", "Lucas", "Priya", "Noah", "Fatima", "Ethan", "Yuki", "Carlos",
         "Amara", "Daniel", "Leila", "Mateo", "Hana", "James", "Zara", "Ravi", "Elena", "Kofi", "Nora", "Arjun",
         "Chloe", "Samir", "Ines", "Tom", "Layla", "Viktor"]
LAST = ["Khan", "Johnson", "Rossi", "Haddad", "Chen", "Silva", "Patel", "Williams", "Al-Mansoori", "Brown",
        "Tanaka", "Garcia", "Okafor", "Miller", "Nasser", "Lopez", "Kim", "Taylor", "Ahmed", "Sharma", "Petrova",
        "Mensah", "Fischer", "Iyer", "Dubois", "Rahman", "Costa", "Walker", "Saleh", "Novak"]
OFFICES = ["Dubai HQ", "London", "Austin", "Singapore"]
PTO_ALLOWANCE = {"Full-Time": 25, "Part-Time": 12, "Contractor": 0}


def employee_rows(today: date | None = None, n: int = 30) -> list[dict]:
    """Deterministic sample employees. Hire dates are relative to today so the demo stays stable."""
    today = today or date.today()
    rng = random.Random(42)
    last = LAST[:]
    rng.shuffle(last)
    dept_names = list(DEPARTMENTS)
    rows = []
    for i in range(n):
        dept = dept_names[i % len(dept_names)]
        titles, lo, hi = DEPARTMENTS[dept]
        etype = rng.choices(["Full-Time", "Part-Time", "Contractor"], weights=[75, 15, 10])[0]
        salary = round(rng.randint(lo, hi) / 500) * 500
        if etype == "Part-Time":
            salary = round(salary * 0.5 / 500) * 500
        allowance = PTO_ALLOWANCE[etype]
        used = rng.randint(0, allowance) if allowance else 0
        first, surname = FIRST[i], last[i]
        rows.append({
            "employee_id": 1001 + i,
            "full_name": f"{first} {surname}",
            "email": f"{first.lower()}.{surname.lower().replace('-', '')}@northstar.demo",
            "department": dept,
            "job_title": rng.choice(titles),
            "employment_type": etype,
            "hire_date": today - timedelta(days=rng.randint(90, 3650)),
            "salary": salary,
            "performance_rating": rng.choice([2, 3, 3, 4, 4, 4, 5, 5]),
            "office_location": rng.choice(OFFICES),
            "pto_days_used": used,
            "pto_days_remaining": allowance - used,
        })
    return rows


HANDBOOK = [
    ("Northstar Analytics - Employee Handbook", [
        "Welcome to Northstar Analytics. This handbook explains the policies that apply to all employees. "
        "It is a fictional document generated for the QuickQuery AI demo.",
        "Contents: Section 1 - Employment Types. Section 2 - Remote Work Policy. "
        "Section 3 - Paid Time Off (PTO). Section 4 - Compensation and Performance Reviews.",
        "Questions about any policy should be sent to the People Team at people@northstar.demo.",
    ]),
    ("Section 1 - Employment Types", [
        "1.1 Full-Time employees work 40 hours per week and are eligible for all benefits described in this handbook.",
        "1.2 Part-Time employees work fewer than 30 hours per week. They receive pro-rated PTO and are not "
        "eligible for full-time remote work.",
        "1.3 Contractors are engaged for a fixed scope of work through a services agreement. Contractors are "
        "not eligible for PTO or for remote work arrangements under this handbook.",
        "1.4 Tenure (continuous service) is measured from the employee's hire date.",
    ]),
    ("Section 2 - Remote Work Policy", [
        "2.1 Northstar supports flexible work. There are two arrangements: Hybrid and Full-Time Remote.",
        "2.2 Full-Time Remote eligibility. An employee qualifies for full-time remote work only if ALL of the "
        "following criteria are met: (a) the employment type is Full-Time; (b) the employee has at least 2 years "
        "(24 months) of continuous service, measured from the hire date; (c) the most recent performance rating "
        "is 4 or higher on the 1-5 scale; and (d) the employee does not work in the Operations or Customer Support "
        "departments, because those roles require on-site presence.",
        "2.3 Hybrid work. Full-Time employees with at least 6 months of service may work remotely up to 2 days "
        "per week with manager approval.",
        "2.4 Approved full-time remote employees receive a one-time home office stipend of USD 500.",
        "2.5 If a remote employee's performance rating falls below 4, the arrangement is reviewed within 30 days.",
    ]),
    ("Section 3 - Paid Time Off (PTO)", [
        "3.1 Annual allowance. Full-Time employees receive 25 PTO days per calendar year. Part-Time employees "
        "receive 12 PTO days per calendar year. Contractors are not eligible for PTO.",
        "3.2 PTO accrues monthly and the current balance is shown in the HR system as PTO days remaining.",
        "3.3 Carry-over. A maximum of 5 unused PTO days may be carried over to the next calendar year. Carried-over "
        "days must be used by March 31, otherwise they are forfeited. Unused days above the 5-day limit are forfeited "
        "on December 31 and are not paid out, except where local law requires it.",
        "3.4 Requests for 3 or more consecutive days must be submitted at least 2 weeks in advance.",
        "3.5 Sick leave is separate from PTO: every employee receives 10 paid sick days per year.",
        "3.6 Public holidays do not count against the PTO allowance.",
    ]),
    ("Section 4 - Compensation and Performance Reviews", [
        "4.1 Performance is reviewed once a year in March using a 1-5 rating scale: 1 = Unsatisfactory, "
        "2 = Needs Improvement, 3 = Meets Expectations, 4 = Exceeds Expectations, 5 = Outstanding.",
        "4.2 Merit increases: a rating of 5 is eligible for up to 8 percent, a rating of 4 for up to 5 percent and "
        "a rating of 3 for up to 2 percent. Ratings of 1 or 2 are not eligible for a merit increase.",
        "4.3 Salary bands are reviewed annually against market data for each department.",
    ]),
]


class _HandbookPDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Northstar Analytics Employee Handbook - Page {self.page_no()}", align="C")


def build_handbook(path: Path = PDF_PATH) -> None:
    pdf = _HandbookPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    for title, paragraphs in HANDBOOK:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(20, 30, 60)
        pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        for p in paragraphs:
            pdf.multi_cell(0, 6, p, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
    pdf.output(str(path))


def build_database(path: Path = DB_PATH) -> None:
    engine = create_engine(f"sqlite:///{path}")
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(employees.insert(), employee_rows())
    engine.dispose()


def ensure_demo_data() -> list[str]:
    """Create missing demo files. Returns the names of files created."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    if not DB_PATH.exists():
        build_database()
        created.append(DB_PATH.name)
    if not PDF_PATH.exists():
        build_handbook()
        created.append(PDF_PATH.name)
    return created


def reset_demo_data() -> list[str]:
    for p in (DB_PATH, PDF_PATH):
        p.unlink(missing_ok=True)
    return ensure_demo_data()


if __name__ == "__main__":
    print("Created:", reset_demo_data())
