from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Model(Base):
    """One row per tuned model - a lookup/reference table replacing the TEST_METRICS dict
    hardcoded in api/model_registry.py. Seeded once from the Final Evaluation results
    (see db/seed_models.py), not written to by the API itself."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    test_precision: Mapped[float] = mapped_column(Float, nullable=False)
    test_recall: Mapped[float] = mapped_column(Float, nullable=False)
    test_f1: Mapped[float] = mapped_column(Float, nullable=False)
    test_pr_auc: Mapped[float] = mapped_column(Float, nullable=False)
    test_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Prediction(Base):
    """One row per /predict call - the audit log. Feature columns mirror
    api/schemas.py::CreditApplication exactly, as plain columns (not a JSON blob) so the
    table supports real SQL aggregation, not just record-by-record lookup."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    revolving_utilization_of_unsecured_lines: Mapped[float] = mapped_column(Float, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_time_30_59_days_past_due_not_worse: Mapped[int] = mapped_column(Integer, nullable=False)
    debt_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    monthly_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    number_of_open_credit_lines_and_loans: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_times_90_days_late: Mapped[int] = mapped_column(Integer, nullable=False)
    number_real_estate_loans_or_lines: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_time_60_89_days_past_due_not_worse: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_dependents: Mapped[float | None] = mapped_column(Float, nullable=True)

    probability_of_default: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[bool] = mapped_column(Boolean, nullable=False)

    model: Mapped["Model"] = relationship(back_populates="predictions")
