import pandas as pd
import pytest

from ml.data.loader import split_data, split_X_y, subsample_for_rbf


class TestSplitXY:
    def test_target_removed_from_X_present_in_y(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "SeriousDlqin2yrs": [0, 1]})
        X, y = split_X_y(df)
        assert "SeriousDlqin2yrs" not in X.columns
        assert y.tolist() == [0, 1]
        assert list(X.columns) == ["a", "b"]


class TestSplitData:
    def test_stratified_split_preserves_class_balance(self):
        # 500 rows, 10% positive - matches the real dataset's rough imbalance shape
        n = 500
        df = pd.DataFrame(
            {
                "feature": range(n),
                "SeriousDlqin2yrs": [1 if i % 10 == 0 else 0 for i in range(n)],
            }
        )
        train_df, test_df = split_data(df, test_size=0.2, random_state=42)

        train_rate = train_df["SeriousDlqin2yrs"].mean()
        test_rate = test_df["SeriousDlqin2yrs"].mean()
        assert train_rate == pytest.approx(0.1, abs=0.02)
        assert test_rate == pytest.approx(0.1, abs=0.02)

    def test_split_sizes_match_test_size(self):
        df = pd.DataFrame({"feature": range(100), "SeriousDlqin2yrs": [0, 1] * 50})
        train_df, test_df = split_data(df, test_size=0.2, random_state=42)
        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_same_random_state_gives_same_split(self):
        df = pd.DataFrame({"feature": range(100), "SeriousDlqin2yrs": [0, 1] * 50})
        train_a, _ = split_data(df, random_state=42)
        train_b, _ = split_data(df, random_state=42)
        assert train_a.index.tolist() == train_b.index.tolist()


class TestSubsampleForRBF:
    def test_returns_requested_size(self):
        df = pd.DataFrame({"feature": range(1000)})
        y = pd.Series([1 if i % 10 == 0 else 0 for i in range(1000)])
        X_sub, y_sub = subsample_for_rbf(df, y, subsample_size=100, random_state=42)
        assert len(X_sub) == 100
        assert len(y_sub) == 100

    def test_subsample_stays_stratified(self):
        df = pd.DataFrame({"feature": range(1000)})
        y = pd.Series([1 if i % 10 == 0 else 0 for i in range(1000)])
        _, y_sub = subsample_for_rbf(df, y, subsample_size=200, random_state=42)
        assert y_sub.mean() == pytest.approx(0.1, abs=0.03)
