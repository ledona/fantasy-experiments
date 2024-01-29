# Historic Daily Fantasy Results Processing
Historic betting and contest data. Used to generate backtest
models that predict for dfs contest winning scores. To generate/refresh
models do the following.

1. Download contest data files from the daily fantasy service.
1. Retrieve detailed data history using _retrieve_hist_
1. transform data into a training dataset using _data_xform_
1. train new models

## Data Retrieval (__retrieve_hist__)
Use _retrieve_hist_ to retrieve contest data from dfs websites.
```
python -m lib.retrieve.retrieve_hist --cache-path /fantasy-archive/betting df-hist-cache/ draftkings \
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

