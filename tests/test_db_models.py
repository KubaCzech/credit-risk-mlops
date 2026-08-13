import pytest
from sqlalchemy.exc import IntegrityError

from db.models import Model, Prediction


class TestModel:
    def test_insert_and_read_back(self, db_session):
        db_session.add(
            Model(
                name="test_model",
                test_precision=0.5,
                test_recall=0.5,
                test_f1=0.5,
                test_pr_auc=0.5,
                test_accuracy=0.5,
                mlflow_run_id="abc123",
            )
        )
        db_session.commit()

        row = db_session.query(Model).filter_by(name="test_model").one()
        assert row.test_pr_auc == 0.5

    def test_name_must_be_unique(self, db_session):
        db_session.add(
            Model(
                name="dup",
                test_precision=0,
                test_recall=0,
                test_f1=0,
                test_pr_auc=0,
                test_accuracy=0,
                mlflow_run_id="x",
            )
        )
        db_session.commit()

        db_session.add(
            Model(
                name="dup",
                test_precision=0,
                test_recall=0,
                test_f1=0,
                test_pr_auc=0,
                test_accuracy=0,
                mlflow_run_id="y",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestPrediction:
    def test_model_id_must_reference_an_existing_model(self, db_session):
        db_session.add(
            Prediction(
                model_id=99999,  # no such model row exists
                revolving_utilization_of_unsecured_lines=0.1,
                age=30,
                number_of_time_30_59_days_past_due_not_worse=0,
                debt_ratio=0.1,
                number_of_open_credit_lines_and_loans=1,
                number_of_times_90_days_late=0,
                number_real_estate_loans_or_lines=0,
                number_of_time_60_89_days_past_due_not_worse=0,
                probability_of_default=0.5,
                prediction=False,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_monthly_income_and_dependents_are_nullable(self, db_session):
        model = Model(
            name="m", test_precision=0, test_recall=0, test_f1=0, test_pr_auc=0, test_accuracy=0, mlflow_run_id="x"
        )
        db_session.add(model)
        db_session.flush()

        db_session.add(
            Prediction(
                model_id=model.id,
                revolving_utilization_of_unsecured_lines=0.1,
                age=30,
                number_of_time_30_59_days_past_due_not_worse=0,
                debt_ratio=0.1,
                monthly_income=None,
                number_of_open_credit_lines_and_loans=1,
                number_of_times_90_days_late=0,
                number_real_estate_loans_or_lines=0,
                number_of_time_60_89_days_past_due_not_worse=0,
                number_of_dependents=None,
                probability_of_default=0.5,
                prediction=False,
            )
        )
        db_session.commit()  # should not raise

    def test_relationship_navigates_to_model(self, db_session):
        model = Model(
            name="m2", test_precision=0, test_recall=0, test_f1=0, test_pr_auc=0, test_accuracy=0, mlflow_run_id="x"
        )
        db_session.add(model)
        db_session.flush()

        pred = Prediction(
            model_id=model.id,
            revolving_utilization_of_unsecured_lines=0.1,
            age=30,
            number_of_time_30_59_days_past_due_not_worse=0,
            debt_ratio=0.1,
            number_of_open_credit_lines_and_loans=1,
            number_of_times_90_days_late=0,
            number_real_estate_loans_or_lines=0,
            number_of_time_60_89_days_past_due_not_worse=0,
            probability_of_default=0.5,
            prediction=False,
        )
        db_session.add(pred)
        db_session.commit()

        assert pred.model.name == "m2"
