from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    """Field names/aliases mirror the raw Kaggle dataset columns exactly - the fitted
    pipelines (cleaning + feature engineering + scaling + model, all one joblib artifact)
    expect those exact column names. MonthlyIncome and NumberOfDependents are optional:
    the pipeline's imputers were built to handle exactly this real-world case."""

    model_config = ConfigDict(populate_by_name=True)

    revolving_utilization_of_unsecured_lines: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines", ge=0, description="Total balance on credit lines / total credit limits"
    )
    age: int = Field(ge=18, description="Age in years")
    number_of_time_30_59_days_past_due_not_worse: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0
    )
    debt_ratio: float = Field(alias="DebtRatio", ge=0)
    monthly_income: float | None = Field(default=None, alias="MonthlyIncome", ge=0)
    number_of_open_credit_lines_and_loans: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0)
    number_of_times_90_days_late: int = Field(alias="NumberOfTimes90DaysLate", ge=0)
    number_real_estate_loans_or_lines: int = Field(alias="NumberRealEstateLoansOrLines", ge=0)
    number_of_time_60_89_days_past_due_not_worse: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0
    )
    number_of_dependents: float | None = Field(default=None, alias="NumberOfDependents", ge=0)


class PredictionRequest(BaseModel):
    application: CreditApplication
    model_name: str = Field(default="xgboost", description="See GET /models for available names")


class PredictionResponse(BaseModel):
    model_used: str
    probability_of_default: float
    prediction: int


class ModelMetrics(BaseModel):
    test_precision: float
    test_recall: float
    test_f1: float
    test_pr_auc: float
    test_accuracy: float


class ModelInfo(BaseModel):
    name: str
    metrics: ModelMetrics


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
