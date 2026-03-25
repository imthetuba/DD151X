from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_split_indices(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    strategy: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    if strategy == "stratified":
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col],
        )
    elif strategy == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y=df[target_col], groups=df[group_col]))
        train_idx = df.index[train_idx]
        test_idx = df.index[test_idx]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return {
        "train_idx": list(map(int, train_idx)),
        "test_idx": list(map(int, test_idx)),
    }
