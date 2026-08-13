import pytest
from sqlalchemy import select

from db.models import Prediction
from ml.monitoring import MIN_SAMPLE_SIZE


class TestHealth:
    def test_health_returns_ok_and_lists_all_8_models(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert len(body["models_loaded"]) == 8


class TestModels:
    def test_lists_8_models_with_metrics(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        body = response.json()
        assert body["default_model"] == "xgboost"
        assert len(body["models"]) == 8
        names = {m["name"] for m in body["models"]}
        assert "xgboost" in names
        assert "mlp" in names

    def test_each_model_has_all_5_metrics(self, client):
        body = client.get("/models").json()
        for model in body["models"]:
            metrics = model["metrics"]
            for key in ["test_precision", "test_recall", "test_f1", "test_pr_auc", "test_accuracy"]:
                assert isinstance(metrics[key], float)


class TestPredict:
    def test_valid_request_returns_prediction(self, client, sample_application):
        response = client.post("/predict", json={"application": sample_application, "model_name": "xgboost"})
        assert response.status_code == 200
        body = response.json()
        assert body["model_used"] == "xgboost"
        assert 0.0 <= body["probability_of_default"] <= 1.0
        assert body["prediction"] in (0, 1)

    def test_known_defaulter_scores_high_probability(self, client, sample_application):
        # sample_application is cs-training.csv's first row, which really did default -
        # not a hardcoded threshold on the model, but a sanity check it isn't inverted/broken
        response = client.post("/predict", json={"application": sample_application, "model_name": "xgboost"})
        assert response.json()["probability_of_default"] > 0.5

    def test_unknown_model_returns_422(self, client, sample_application):
        response = client.post("/predict", json={"application": sample_application, "model_name": "not_a_real_model"})
        assert response.status_code == 422

    def test_age_below_18_returns_422(self, client, sample_application):
        sample_application["age"] = 10
        response = client.post("/predict", json={"application": sample_application})
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client, sample_application):
        del sample_application["DebtRatio"]
        response = client.post("/predict", json={"application": sample_application})
        assert response.status_code == 422

    def test_optional_fields_can_be_omitted(self, client, sample_application):
        del sample_application["MonthlyIncome"]
        del sample_application["NumberOfDependents"]
        response = client.post("/predict", json={"application": sample_application, "model_name": "naive_bayes"})
        assert response.status_code == 200

    def test_prediction_is_written_to_the_database(self, client, sample_application, seeded_db):
        client.post("/predict", json={"application": sample_application, "model_name": "xgboost"})

        rows = seeded_db.execute(select(Prediction)).scalars().all()
        assert len(rows) == 1
        assert rows[0].age == sample_application["age"]
        assert rows[0].prediction is True

    def test_each_model_name_is_individually_usable(self, client, sample_application):
        for model_name in [
            "logistic_regression",
            "linear_svm",
            "rbf_svm",
            "random_forest",
            "decision_tree",
            "xgboost",
            "naive_bayes",
            "mlp",
        ]:
            response = client.post("/predict", json={"application": sample_application, "model_name": model_name})
            assert response.status_code == 200, f"{model_name} failed: {response.text}"


class TestPredictExplain:
    # See ml/explainability.py's module docstring for why each model lands in this
    # particular output_space - confirmed empirically, not assumed.
    EXPECTED_OUTPUT_SPACE = {
        "logistic_regression": "log_odds",
        "linear_svm": "log_odds",
        "rbf_svm": "margin",
        "random_forest": "probability",
        "decision_tree": "probability",
        "xgboost": "log_odds",
        "naive_bayes": "probability",
        "mlp": "probability",
    }

    def test_valid_request_returns_16_ranked_contributions(self, client, sample_application):
        response = client.post("/predict/explain", json={"application": sample_application, "model_name": "xgboost"})
        assert response.status_code == 200
        body = response.json()
        assert body["model_used"] == "xgboost"
        assert body["output_space"] == "log_odds"
        assert len(body["contributions"]) == 16

        magnitudes = [abs(c["shap_value"]) for c in body["contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_unknown_model_returns_422(self, client, sample_application):
        response = client.post(
            "/predict/explain", json={"application": sample_application, "model_name": "not_a_real_model"}
        )
        assert response.status_code == 422

    def test_each_model_name_is_individually_usable_and_matches_expected_output_space(self, client, sample_application):
        for model_name, expected_space in self.EXPECTED_OUTPUT_SPACE.items():
            response = client.post(
                "/predict/explain", json={"application": sample_application, "model_name": model_name}
            )
            assert response.status_code == 200, f"{model_name} failed: {response.text}"
            assert response.json()["output_space"] == expected_space, model_name

    def test_probability_space_explanations_are_additive(self, client, sample_application):
        # SHAP's efficiency axiom: base_value + sum(contributions) == the explained function's
        # own output for that instance - for probability-space models, that's exactly
        # probability_of_default, so it's checkable here without duplicating ml/explainability.py.
        for model_name in ["random_forest", "decision_tree", "naive_bayes", "mlp"]:
            predict_response = client.post(
                "/predict", json={"application": sample_application, "model_name": model_name}
            )
            explain_response = client.post(
                "/predict/explain", json={"application": sample_application, "model_name": model_name}
            )
            probability = predict_response.json()["probability_of_default"]
            body = explain_response.json()
            reconstructed = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
            assert reconstructed == pytest.approx(probability, abs=1e-4), model_name


class TestMonitoringDrift:
    def test_below_min_sample_size_reports_insufficient(self, client, sample_application, seeded_db):
        client.post("/predict", json={"application": sample_application, "model_name": "xgboost"})

        response = client.get("/monitoring/drift", params={"model_name": "xgboost"})
        assert response.status_code == 200
        body = response.json()
        assert body["n_predictions"] == 1
        assert body["sufficient_sample"] is False
        assert body["score_psi"] is None
        assert body["feature_psi"] == {}

    def test_identical_repeated_requests_register_as_significant_drift(self, client, sample_application, seeded_db):
        # Real production traffic has variance; MIN_SAMPLE_SIZE copies of the exact same
        # row is about as far from the train reference's shape as traffic can get - a cheap,
        # deterministic way to exercise the "yes, this fires" path without needing real
        # varied traffic in a test.
        for _ in range(MIN_SAMPLE_SIZE + 5):
            client.post("/predict", json={"application": sample_application, "model_name": "xgboost"})

        response = client.get("/monitoring/drift", params={"model_name": "xgboost"})
        assert response.status_code == 200
        body = response.json()
        assert body["sufficient_sample"] is True
        assert body["score_psi"]["status"] == "significant_shift"
        # every raw feature present, "age" specifically called out - it's the one field
        # with no explicit Pydantic alias (its Kaggle name is already lowercase "age"),
        # which once silently dropped it from the Kaggle-name -> DB-attribute mapping
        assert set(body["feature_psi"]) == set(sample_application.keys())
        assert "age" in body["feature_psi"]

    def test_unknown_model_returns_422(self, client, sample_application):
        response = client.get("/monitoring/drift", params={"model_name": "not_a_real_model"})
        assert response.status_code == 422


class TestMetrics:
    def test_metrics_endpoint_exposes_prometheus_text_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text or "http_request_duration" in response.text
