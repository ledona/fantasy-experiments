# Historic Daily Fantasy Results Processing
Historic betting and contest data. Used to generate backtest
models that predict for dfs contest winning scores

## Data Retrieval
Use _retrieve_hist_ to retrieve contest data from dfs websites.
```
python -m lib.retrieve_hist --cache-path /fantasy-archive/betting df-hist-cache/ draftkings \
   [--sports nfl] [--cache-only] [--start-date 20201001] [--end-date 20210101]
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

### To retrieve historic betting data
1. download betting history from services and save in the history file directory (e.g. /
   fantasy-archive/betting). Make sure that the file names match the format of the previous
   files for the service
