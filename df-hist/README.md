# Historic Daily Fantasy Results Processing
Historic betting and contest data. Used to generate backtest
models that predict for dfs contest winning scores. To generate/refresh
models do the following.

1. Download contest data files from the daily fantasy service.
1. Retrieve detailed data history using _retrieve_hist_
1. transform data into a training dataset using _data_xform_
1. train new models

## Data Retrieval (__retrieve_hist__)
Use _retrieve_hist_ to retrieve contest data from dfs websites. This is not done from
within the docker container, easier to just run it in the native local environment
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
3. Run the retrieval. Update the paths before running. The cache folder is where the retrieval process will write cache files, the export folder is where the files downloaded from the fantasy service accounts (with past betting activity) are located. See the _launch.json_ entry for more argument examples
```
python -m lib.retrieve.retrieve_hist \
   --cache-path _PATH_TO_CACHE_FOLDER_ \
   --history-file-path _PATH_TO_DFS_EXPORT_FILE_FOLDER_ \
   draftkings \
   [--sports nfl] [--cache-only] [--start-date 20201001] [--end-date 20210101]
```


## Create model training data (__data_xform__)
First make sure that _lib/data_cfg.py_ is up to date, then run _data_xform_. The following example
call to _data_xform_ will create datasets for nfl and mlb using defaults for all settings.
Default settings will write training data into the _data_ folder.
```
python -m lib.xform.cli nfl mlb
```

## Create models
Review/update _model_cfg.json_ , then run the following. Models will be written to the
_models_ directory. Evaluation results will be written to a timestamp named file in 
_eval\_results_ .
```
python -m lib.modeling.cli nfl nba
```


## OUTDATED!!!
### setup the environment
```
# EDIT dot_envrc then
source dot_envrc
# RUN THE FOLLOWING ON THE CHROMEDRIVER BINARY...
perl -pi -e 's/cdc/dog/g' /path/to/chromedriver

# then use the Model Export/Testing notebooks
# install automl + onnx
sudo apt-get install libprotobuf-dev protobuf-compiler swig
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
pip install -r requirements
```

