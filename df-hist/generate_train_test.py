from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from fantasy_py import ContestStyle


COLS_TO_IGNORE = {
    'date', 'style', 'type', 'link', 'entries', 'slate_id',
    'top_score', 'last_winning_score',
}


def load_csv(sport, service, style: ContestStyle, contest_type) -> pd.DataFrame:
    filename = f"{sport}-{service}-{style.name}-{contest_type.NAME}.csv"
    print(f"loading {filename=}")

    df = pd.read_csv(filename)
    print(f"{len(df)} rows of data loaded")
    nan_slate_rows = len(df.query('slate_id.isnull()'))
    nan_best_score_rows = len(df.query('`best-possible-score`.isnull()'))
    if nan_slate_rows > 0 or nan_best_score_rows > 0:
        print(f"dropping {nan_slate_rows + nan_best_score_rows} rows due to {nan_slate_rows=} {nan_best_score_rows=}")
        df = df.dropna()
    return df


def generate_train_test(df, train_size: float = .5,
                        random_state: Optional[int] = None,
                        model_cols: Optional[set[str]] = None) -> Optional[tuple]:
    """
    create regression train test data
    model_cols - if none then use all available columns
    return (X-train, X-test y-top-train, y-top-test, y-last-win-train, y-last-win-test)
    """
    x_cols = []
    assert (model_cols is None) or model_cols <= set(df.columns), \
        "Requested model columns not a subset of available data columns"
    for col in df.columns:
        if col in COLS_TO_IGNORE:
            continue
        assert col[0] == '(' or col.startswith('team') or col == 'best-possible-score', \
            f"Unexpected data column named '{col}'"

        if (model_cols is None) or col in model_cols:
            x_cols.append(col)

    X = df[x_cols]
    if len(X) == 0:
        return None
    # display(X)
    y_top = df.top_score
    # display(y_top)
    y_last_win = df.last_winning_score
    # display(y_last_win)

    return train_test_split(X, y_top, y_last_win,
                            random_state=random_state,
                            train_size=train_size)
