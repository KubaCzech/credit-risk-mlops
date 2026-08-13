import pandas as pd

from ml.features.engineering import DelinquencyAggregator, InteractionFeatures


class TestDelinquencyAggregator:
    def test_sums_the_three_columns(self):
        df = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 1],
                "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 0],
                "NumberOfTimes90DaysLate": [0, 0, 3],
            }
        )
        result = DelinquencyAggregator().fit(df).transform(df)
        assert result["total_delinquency"].tolist() == [0, 3, 4]

    def test_has_any_delinquency_flag_matches_total(self):
        df = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [0, 1],
                "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
                "NumberOfTimes90DaysLate": [0, 0],
            }
        )
        result = DelinquencyAggregator().fit(df).transform(df)
        assert result["has_any_delinquency"].tolist() == [0, 1]

    def test_original_columns_untouched(self):
        df = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [2],
                "NumberOfTime60-89DaysPastDueNotWorse": [0],
                "NumberOfTimes90DaysLate": [0],
            }
        )
        result = DelinquencyAggregator().fit(df).transform(df)
        assert result["NumberOfTime30-59DaysPastDueNotWorse"].tolist() == [2]


class TestInteractionFeatures:
    def test_utilization_x_delinquency_is_the_product(self):
        df = pd.DataFrame(
            {
                "RevolvingUtilizationOfUnsecuredLines": [0.5, 2.0],
                "total_delinquency": [4, 0],
                "MonthlyIncome": [1000.0, 5000.0],
                "NumberOfDependents": [0.0, 3.0],
            }
        )
        result = InteractionFeatures().fit(df).transform(df)
        assert result["utilization_x_delinquency"].tolist() == [2.0, 0.0]

    def test_income_per_dependent_adds_one_to_avoid_division_by_zero(self):
        df = pd.DataFrame(
            {
                "RevolvingUtilizationOfUnsecuredLines": [0.0],
                "total_delinquency": [0],
                "MonthlyIncome": [1000.0],
                "NumberOfDependents": [0.0],  # zero dependents - would divide by zero without +1
            }
        )
        result = InteractionFeatures().fit(df).transform(df)
        assert result["income_per_dependent"].tolist() == [1000.0]

    def test_income_per_dependent_scales_down_with_more_dependents(self):
        df = pd.DataFrame(
            {
                "RevolvingUtilizationOfUnsecuredLines": [0.0, 0.0],
                "total_delinquency": [0, 0],
                "MonthlyIncome": [1000.0, 1000.0],
                "NumberOfDependents": [0.0, 4.0],
            }
        )
        result = InteractionFeatures().fit(df).transform(df)
        income_0_dep, income_4_dep = result["income_per_dependent"]
        assert income_4_dep < income_0_dep
