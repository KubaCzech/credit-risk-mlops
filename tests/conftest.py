import os

# Must be set before db.config/db.session are imported by anything (including indirectly,
# e.g. api.main) - both read DATABASE_URL once, at import time. Setting it here, first
# thing in conftest.py, guarantees it wins: pytest imports conftest.py before collecting
# any test module. Points at a dedicated database so tests never touch dev data.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://credit_risk:credit_risk@localhost:5432/credit_risk_test")

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from db.models import Base
from db.session import SessionLocal, engine


@pytest.fixture(scope="session", autouse=True)
def _test_schema():
    try:
        Base.metadata.create_all(engine)
    except OperationalError as e:
        pytest.skip(f"credit_risk_test database not reachable ({e}) - see README Tests section")
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session():
    """One session per test, tables truncated after - simpler and more robust than
    transaction-rollback isolation given the code under test (api/main.py) calls
    session.commit() itself, which would end an outer transaction early."""
    session = SessionLocal()
    yield session
    session.close()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE predictions RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE models RESTART IDENTITY CASCADE"))


@pytest.fixture
def seeded_db(db_session):
    """models table populated from the same static snapshot production seeds from."""
    from db.seed_models import seed_models

    seed_models()
    return db_session


@pytest.fixture
def client(seeded_db):
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_application() -> dict:
    """The first row of cs-training.csv, verbatim - a known-good, realistic payload
    (SeriousDlqin2yrs=1 in the source data, i.e. this person really did default)."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.766126609,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.802982129,
        "MonthlyIncome": 9120,
        "NumberOfOpenCreditLinesAndLoans": 13,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 6,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }
