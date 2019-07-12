# MLB

## Setup
DB=mlb-2019.db
DATES="2019-06-01 2019-06-12 2019-06-16"
OUT=mlb_lineup_opt

## Pick a service
SERVICE=draftkings|fanduel|yahoo

## Style
STYLE=CLASSIC|SHOWDOWN

## Run the command
cmd="lineup_hp_optimizer.sc --slack --search_bayes_fail_fast $DB $SERVICE $DATES
--generations_range 50 2000 --population_range 100 2000
--offspring_range 100 2000 --runs_range 1 3 --search_method bayes
--search_iterations 25 --cache_dir lineup_cache
--crossover_prob_range .1 .9 --mut_prob_range .1 .9 
--mut_full_swapdrop_range .1 .9 --mut_filling_swapadd_range .1 .9
--mut_max_pct_range .1 .9
--outfile $OUT.$SERVICE.$STYLE.opt --style $STYLE"
