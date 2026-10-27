# Historic Daily Fantasy Results Processing
Historic betting and contest data. Used to generate backtest
models that predict for dfs contest winning scores. To generate/refresh
models do the following.

1. Download contest data files from the daily fantasy service.
1. Retrieve detailed data history using _retrieve_hist_
1. transform data into a training dataset using _data_xform_
1. train new winning score prediction models

## Data Retrieval (__retrieve_hist__)
Use _retrieve_hist_ to retrieve contest data from dfs websites. __This is not done from
within the docker container__, instead it is run in the native local environment to facilitate selenium orchestration. The following instructions assume this is run on Windows

1. Make sure that the following windows environment variables are set (use `echo $env:VARIABLE`)
   - FANTASY_IDRIVE_HOME : windows path to the IDRIVE cloud fantasy dir
2. Create a venv to run the retrieval in by running the following command on the commandline in the local environment (again, not in the container).
```
uv venv venv-retrieve-hist
python -m venv venv-retrieve-hist
.\venv-retrieve-hist\Scripts\activate
uv pip install pandas selenium beautifulsoup4 tqdm
```
3. Start chrome with a debugging port. On windows run the following from powershell
```
Start-Process "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList @("--remote-debugging-port=9222", "--user-data-dir=$env:USERPROFILE\working\fantasy-experiments\df_hist\chrome-user-data")
```
To verify that chrome is running correctly try navigating to `http://127.0.0.1:9222/json/version`
4. Run the retrieval. If running in vscode be sure to set the python environment before attempting to run. Then update the paths before running. The cache folder is where the retrieval process will write cache files, the export folder is where the files downloaded from the fantasy service accounts (with past betting activity) are located. See the _launch.json_ entry for more argument examples. Draftkings updates will use the most recent user data file found.
```
python -m lib.retrieve.retrieve_hist \
   --cache-path _PATH_TO_CACHE_FOLDER_ \
   --history-file-path _PATH_TO_DFS_EXPORT_FILE_FOLDER_ \
   --skip-entry-filepath _PATH_TO_SKIP_ENTRY_TEXTFILE \
   draftkings \
   [--sports nfl] [--cache-only] [--start-date 20201001] [--end-date 20210101]
```
5. Selenium navigation (i.e. the python script) may fail regularly. There are 2 things to try for getting more reliablility and having the script run longer before stopping. a) Resize/Zoom the browser so that the entire navigational UI is visible. b) Use the --fail_restarts cli arg to retry the retrieval a couple times...

## Create model training data (__data_xform__)
First make sure that _lib/data_cfg.py_ is up to date, then run _data_xform_.

If data retrieval (the previous section) was not last run on all data (no constraints on date, sport, etc) then run it again now with no constraints to ensure that all betting data is ready for transformation.

### Clean xform run
Attempt a 'clean' first run and examine the results. For this run use settings that will surface any data issues but will not fail fast. See the help
for the xform program for information on other important parameters like the cache and output directory destinations.

```
# full clean/test run to seed the cache, this should process all data and for any item where there are missing inference features, the xform features will be na
python -m lib.xform.cli --max_missing_infer_cols 0
```

Surfaced data issues will look like holes in the xform output for missing slates or missing features. Missing slates are typically due to missing historic slate data (maybe fixable if historic data can be pulled) or team name/abbreviation inconsistencies (fixable with team abbr aliasing). The list of missing slates is logged as warnings at the very start of every run.

Missing features can be caused by unmatched slates (see previous item for fixes), failures predicting player/team performance due to missing data (e.g. missing odds data), other failures in xform feature generation. Examine the log to identify what is going on. If features are missing for predicting player/team performance see how many features are missing and use this to set a higher max missing infer cols threshold.

### Rerunning xform
Based on what is learned in the clean run, set the missing inf parameter to something that will tolerate the expected # of missing inf features for the sport. And for the final run set
the program to fail fast to lock things in and surface anything unexpected.
```
# rerun/final form with fail fast and final missing infer cols cap
python -m lib.xform.cli --max_missing_infer_cols N --fail_on_impute_error
```

### clean the scoring cache files
If slate scoring data needs to be recalculated (e.g. rational lineup param changes or other slate scoring pipeline updates),
remove ```*slate.score.json``` from the xform output directory.

## Create models
Review/update _model_cfg.json_ then run the following. Models will be written to the _./models_ directory by default. Use _--model_path_ to write models to a different directory. Evaluation results will be written to a timestamp named file in 
_eval\_results_ .
```
python -m lib.modeling.cli nfl nba \
   --model_path /fantasy-isync/fantasy-dfs-hist/models/2026.10 \
   --results_path /fantasy-isync/fantasy-dfs-hist/eval_results
   --data_path /fantasy-isync/fantasy-dfs-hist/data
   --framework regchain_tree
```
## Update Environment
Backtesting uses models in the directory at the environment variable _FANTASY_BACKTEST_WINSCORE_MODELS_PATH_. Make sure to update it
so that new models are used.