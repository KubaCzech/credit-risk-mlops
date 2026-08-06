from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import INDEX_COL, RANDOM_STATE, RAW_DATA_PATH, RBF_SVM_SUBSAMPLE_SIZE, TARGET_COL, TEST_SIZE


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Kaggle CSV, using its row-id column as the DataFrame index."""
    return pd.read_csv(path, index_col=INDEX_COL)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on target_col; both splits keep the target column."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df, test_df


def load_and_split(
    path: Path = RAW_DATA_PATH,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw_data(path)
    return split_data(df, target_col, test_size, random_state)


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]), df[target_col]


def subsample_for_rbf(
    X: pd.DataFrame,
    y: pd.Series,
    subsample_size: int = RBF_SVM_SUBSAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample used everywhere an RBF SVM is fit - see RBF_SVM_SUBSAMPLE_SIZE
    in config.py for why (O(n^2-n^3) training cost makes the full train set intractable)."""
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=subsample_size, stratify=y, random_state=random_state)
    return X_sub, y_sub
