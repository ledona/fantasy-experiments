# Baseline backtesting to test code and get a starting point for past betting
All testing uses the genetic algorithm

python -O scripts/backtest.sc --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json full_test.fd.json --starting_bankroll 25 mlb-2019.db fanduel SD_gpp_25 C_gpp_25 C_5050_1 SD_5050_1
python -O scripts/backtest.sc --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json full_test.dk.json --starting_bankroll 25 mlb-2019.db draftkings SD_gpp_25 C_gpp_25 C_5050_1 SD_5050_1
python -O scripts/backtest.sc --slack --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json full_test.y.json --starting_bankroll 25 mlb-2019.db yahoo C_gpp_25 C_5050_2
