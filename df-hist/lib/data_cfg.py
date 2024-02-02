import os
from datetime import date


_FANTASY_HOME = os.environ["FANTASY_HOME"]


SPORT_CFGS = {
    "mlb": {
        "min_date": date(2019, 1, 1),
        "max_date": date(2024, 1, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "mlb_hist_20082023.scored.db"),
        "cost_pos_drop": {"DH", "RP"},
        "cost_pos_rename": {"SP": "P"},
    },
    "nfl": {
        "min_date": date(2020, 1, 12),  # no NFL dfs slates before this date
        "max_date": date(2023, 4, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nfl_hist_2009-2022.scored.db"),
    },
    "nba": {
        "min_date": {None: date(2017, 8, 1), "yahoo": date(2020, 8, 1)},
        "max_date": date(2023, 8, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nba_hist_20082009-20222023.scored.db"),
    },
    "nhl": {
        "min_date": {
            "draftkings": date(2019, 10, 9),  # dk changed scoring formula for nhl
            "fanduel": date(2019, 8, 1),  # fd missing positional data prior to 2019 season
            None: date(2017, 8, 1),
        },
        "max_date": date(2023, 4, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nhl_hist_20072008-20222023.scored.db"),
        "cost_pos_rename": {"LW": "W", "RW": "W"},
    },
    "lol": {
        "db_filename": os.path.join(_FANTASY_HOME, "lol_hist_2014-2022.scored.db"),
        "min_date": date(2020, 1, 1),
        "max_date": date(2023, 1, 1),
        "services": ["draftkings", "fanduel"],
    },
}
