import pytest
from pydantic import ValidationError

from api.schemas import CreditApplication, PredictionRequest


def _valid_payload(**overrides) -> dict:
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 30,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
    }
    payload.update(overrides)
    return payload


class TestCreditApplication:
    def test_valid_payload_parses(self):
        app = CreditApplication.model_validate(_valid_payload())
        assert app.age == 30
        assert app.monthly_income is None

    def test_hyphenated_alias_maps_to_correct_field(self):
        app = CreditApplication.model_validate(_valid_payload(**{"NumberOfTime30-59DaysPastDueNotWorse": 3}))
        assert app.number_of_time_30_59_days_past_due_not_worse == 3

    def test_age_below_18_rejected(self):
        with pytest.raises(ValidationError):
            CreditApplication.model_validate(_valid_payload(age=17))

    def test_negative_debt_ratio_rejected(self):
        with pytest.raises(ValidationError):
            CreditApplication.model_validate(_valid_payload(DebtRatio=-1))

    def test_monthly_income_and_dependents_are_optional(self):
        app = CreditApplication.model_validate(_valid_payload())
        assert app.monthly_income is None
        assert app.number_of_dependents is None

    def test_monthly_income_can_be_provided(self):
        app = CreditApplication.model_validate(_valid_payload(MonthlyIncome=5000))
        assert app.monthly_income == 5000

    def test_missing_required_field_rejected(self):
        payload = _valid_payload()
        del payload["DebtRatio"]
        with pytest.raises(ValidationError):
            CreditApplication.model_validate(payload)


class TestPredictionRequest:
    def test_model_name_defaults_to_xgboost(self):
        request = PredictionRequest.model_validate({"application": _valid_payload()})
        assert request.model_name == "xgboost"

    def test_model_name_can_be_overridden(self):
        request = PredictionRequest.model_validate({"application": _valid_payload(), "model_name": "naive_bayes"})
        assert request.model_name == "naive_bayes"
