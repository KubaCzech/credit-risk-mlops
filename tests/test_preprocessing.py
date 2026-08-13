import numpy as np
import pandas as pd

from ml.data.preprocessing import (
    AgeCleaner,
    DelinquencySentinelHandler,
    DependentsImputer,
    MonthlyIncomeImputer,
    OutlierCapper,
)


class TestDelinquencySentinelHandler:
    def test_sentinel_values_replaced_with_train_median(self):
        df = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 98, 4],
                "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 98, 0],
                "NumberOfTimes90DaysLate": [0, 0, 96, 0],
            }
        )
        handler = DelinquencySentinelHandler().fit(df)
        result = handler.transform(df)

        # median of the 3 non-sentinel rows (0, 2, 4) is 2
        assert result.loc[2, "NumberOfTime30-59DaysPastDueNotWorse"] == 2
        assert result["has_delinquency_sentinel"].tolist() == [0, 0, 1, 0]

    def test_flag_set_if_any_of_the_three_columns_is_sentinel(self):
        df = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [0, 98],
                "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
                "NumberOfTimes90DaysLate": [0, 0],
            }
        )
        result = DelinquencySentinelHandler().fit(df).transform(df)
        assert result["has_delinquency_sentinel"].tolist() == [0, 1]

    def test_median_computed_on_train_reused_unchanged_on_test(self):
        train = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 4],
                "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0],
                "NumberOfTimes90DaysLate": [0, 0, 0],
            }
        )
        handler = DelinquencySentinelHandler().fit(train)

        test = pd.DataFrame(
            {
                "NumberOfTime30-59DaysPastDueNotWorse": [98],
                "NumberOfTime60-89DaysPastDueNotWorse": [98],
                "NumberOfTimes90DaysLate": [98],
            }
        )
        result = handler.transform(test)
        # train median (2), not anything derived from the test row itself
        assert result.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"] == 2


class TestMonthlyIncomeImputer:
    def test_missing_filled_with_train_median_and_flagged(self):
        train = pd.DataFrame({"MonthlyIncome": [1000.0, 3000.0, np.nan]})
        imputer = MonthlyIncomeImputer().fit(train)
        result = imputer.transform(train)

        assert result["MonthlyIncome"].tolist() == [1000.0, 3000.0, 2000.0]
        assert result["MonthlyIncome_was_missing"].tolist() == [0, 0, 1]

    def test_no_missing_values_flag_stays_zero(self):
        train = pd.DataFrame({"MonthlyIncome": [1000.0, 2000.0]})
        result = MonthlyIncomeImputer().fit(train).transform(train)
        assert result["MonthlyIncome_was_missing"].tolist() == [0, 0]


class TestDependentsImputer:
    def test_missing_filled_with_train_median_no_flag_column(self):
        train = pd.DataFrame({"NumberOfDependents": [0.0, 2.0, np.nan]})
        result = DependentsImputer().fit(train).transform(train)

        assert result["NumberOfDependents"].tolist() == [0.0, 2.0, 1.0]
        assert "NumberOfDependents_was_missing" not in result.columns


class TestAgeCleaner:
    def test_clips_below_minimum_leaves_rest_untouched(self):
        df = pd.DataFrame({"age": [0, 17, 18, 45]})
        result = AgeCleaner().fit(df).transform(df)
        assert result["age"].tolist() == [18, 18, 18, 45]


class TestOutlierCapper:
    def test_test_set_capped_at_train_derived_threshold(self):
        train = pd.DataFrame({"DebtRatio": list(range(1, 101))})  # 1..100, 99th pct = 99.01
        capper = OutlierCapper(cols=["DebtRatio"], upper_quantile=0.99).fit(train)

        test = pd.DataFrame({"DebtRatio": [5000.0]})
        result = capper.transform(test)
        # capped at the train-fitted bound, not left at its own extreme value
        assert result.loc[0, "DebtRatio"] == capper.upper_bounds_["DebtRatio"]
        assert result.loc[0, "DebtRatio"] < 5000.0

    def test_values_below_threshold_unaffected(self):
        train = pd.DataFrame({"DebtRatio": list(range(1, 101))})
        capper = OutlierCapper(cols=["DebtRatio"], upper_quantile=0.99).fit(train)

        test = pd.DataFrame({"DebtRatio": [0.5]})
        result = capper.transform(test)
        assert result.loc[0, "DebtRatio"] == 0.5
