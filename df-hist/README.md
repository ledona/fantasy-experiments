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
within the docker container__, instead it is run in the native local environment to facilitate selenium orchestration.

1. Create a venv to run the retrieval in by running the following command on the commandline in the local environment (again, not in the container).
```
python -m venv venv-retrieve-hist
.\venv-retrieve-hist\Scripts\activate
pip install pandas selenium beautifulsoup4 tqdm
```
2. Start chrome with a debugging port. On windows run the following from powershell.
```
Start-Process "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList "--remote-debugging-port=9222"
```
3. Run the retrieval. Update the paths before running. The cache folder is where the retrieval process will write cache files, the export folder is where the files downloaded from the fantasy service accounts (with past betting activity) are located. See the _launch.json_ entry for more argument examples. Draftkings updates will use the most recent user data file found.
```
python -m lib.retrieve.retrieve_hist \
   --cache-path _PATH_TO_CACHE_FOLDER_ \
   --history-file-path _PATH_TO_DFS_EXPORT_FILE_FOLDER_ \
   draftkings \
   [--sports nfl] [--cache-only] [--start-date 20201001] [--end-date 20210101]
```
4. (optional) If debugging/running under VSCODE
   a. To update vscode to use the new interpreter, _SHIFT-CTRL-P_, then _Python:Select Interpreter_, then navigate to the python exe in the new venv
   b. The VSCODE launch configuration uses the environment variable _FANTASY_IDRIVE_HOME_. Set this in the environment that the process will be run from

## Create model training data (__data_xform__)
First make sure that _lib/data_cfg.py_ is up to date, then run _data_xform_. The following example
call to _data_xform_ will create datasets for nfl and mlb using defaults for all settings.
Default settings will read betting data from _/fantasy-isync/fantasy-dfs-hist/betting_ and write training data to _/fantasy-isync/fantasy-dfs-hist/data_.
```
python -m lib.xform.cli
```
After data retrieval testing/updating it may be needed to run a full data retrieval process (no constraints on date, sport, etc) in no_cache mode, to ensure that betting data is ready for transformation.

## Create models
Review/update _model_cfg.json_ then run the following. Models will be written to the _./models_ directory by default. Use _--model_path_ to write models to a different directory. Evaluation results will be written to a timestamp named file in 
_eval\_results_ .
```
python -m lib.modeling.cli nfl nba \
   --model_path /fantasy-isync/fantasy-dfs-hist/models/2024.04 \
   --results_path /fantasy-isync/fantasy-dfs-hist/eval_results
   --data_path /fantasy-isync/fantasy-dfs-hist/data
   --framework reg_chain tpot
```
## Update Environment
Backtesting uses models in the directory at the environment variable _FANTASY_BACKTEST_WINSCORE_MODELS_PATH_. Make sure to update it
so that new models are used.