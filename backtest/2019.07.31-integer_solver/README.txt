all tests are for MLB

backtest.sc --slack --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json mi_full_test.dk.json --starting_bankroll 25 mlb-2019.db draftkings SD_gpp_25 C_gpp_25 C_5050_1 SD_5050_1

backtest.sc --slack --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json mi_full_test.fd.json --starting_bankroll 25 mlb-2019.db fanduel SD_gpp_25 C_gpp_25 C_5050_1 SD_5050_1

backtest.sc --slack --cache_dir ./backtest_cache/ --start_game 25 --end_game 100 --season 2019 --dump_json mi_full_test.y.json --starting_bankroll 25 mlb-2019.db yahoo C_gpp_25 C_5050_1

