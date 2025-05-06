import os
from datetime import date


_FANTASY_HOME = os.environ["FANTASY_HOME"]


SPORT_CFGS = {
    "mlb": {
        "min_date": date(2019, 1, 1),
        "max_date": date(2025, 1, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "mlb.hist.2008-2024.scored.db"),
        "cost_pos_drop": {"DH", "RP"},
        "cost_pos_rename": {"SP": "P"},
    },
    "nfl": {
        "min_date": date(2020, 1, 12),  # no NFL dfs slates before this date
        "max_date": date(2025, 4, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nfl.hist.2009-2024.scored.db"),
    },
    "nba": {
        "min_date": {None: date(2017, 8, 1), "yahoo": date(2020, 8, 1)},
        "max_date": date(2025, 8, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nba.hist.20082009-20232024.scored.db"),
    },
    "nhl": {
        "min_date": {
            "draftkings": date(2019, 10, 9),  # dk changed scoring formula for nhl
            "fanduel": date(2019, 8, 1),  # fd missing positional data prior to 2019 season
            None: date(2017, 8, 1),
        },
        "max_date": date(2025, 4, 1),
        "db_filename": os.path.join(_FANTASY_HOME, "nhl.hist.20072008-20232024.scored.db"),
        "cost_pos_rename": {"LW": "W", "RW": "W"},
    },
    "lol": {
        "db_filename": os.path.join(_FANTASY_HOME, "lol.hist.2014-2024.scored.db"),
        "min_date": date(2020, 1, 1),
        "max_date": date(2025, 1, 1),
        "services": ["draftkings", "fanduel"],
    },
}
