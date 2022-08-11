from typing import Optional
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from fantasy_py import ContestStyle


COLS_TO_IGNORE = {
    'date', 'style', 'type', 'link', 'entries', 'slate_id',
    'top_score', 'last_winning_score',
}


def load_csv(sport, service, style: ContestStyle, contest_type, data_folder=".") -> pd.DataFrame:
    filename = f"{sport}-{service}-{style.name}-{contest_type.NAME}.csv"
    filepath = os.path.join(data_folder, filename)
    print(f"loading {filepath=}")

    df = pd.read_csv(filepath)
    print(f"{len(df)} rows of data loaded")
    nan_slate_rows = len(df.query('slate_id.isnull()'))
    nan_best_score_rows = len(df.query('`best-possible-score`.isnull()'))
    if nan_slate_rows > 0 or nan_best_score_rows > 0:
        df = df.dropna()
        print(f"dropped {nan_slate_rows + nan_best_score_rows} rows due to {nan_slate_rows=} {nan_best_score_rows=}. {len(df)=}")
    return df


def generate_train_test(df: pd.DataFrame, train_size: float = .5,
                        random_state: Optional[int] = None,
                        model_cols: Optional[set[str]] = None) -> Optional[tuple]:
    """
    create regression train test data
    model_cols - if none then use all available columns
    return (X-train, X-test y-top-train, y-top-test, y-last-win-train, y-last-win-test)
    """
    x_cols = []
    assert (model_cols is None) or \
        (type(model_cols) == set and model_cols <= set(df.columns)) or \
        (type(model_cols) == str and model_cols in set(df.columns)), \
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
    y_top = df['top_score']
    # display(y_top)
    y_last_win = df['last_winning_score']
    # display(y_last_win)

    try:
        sample_data = train_test_split(X, y_top, y_last_win,
                                       random_state=random_state,
                                       train_size=train_size)
    except ValueError as ex:
        print(f"generate_train_test_split:: Error generating train test split: {ex}")
        sample_data = None
        
    return sample_data
