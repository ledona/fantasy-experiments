## Model Creation
To create, archive and use new predictive models perform the following steps

1. Make sure that the fantasy environment is successfully installed and usable, and the 
database files containing raw and calculated stats are in $FANTASY_HOME.
1. Create a new model folder. Easiest to copy the most recent model folder and rename.
1. Create/update the data export scripts and model training json files. The training files are json dicts with the following structure (refer to previous files for concrete examples):
```
{
  "global_defaults": {
    # dict with default training parameters for all models
    # defined in this configuration file
  },
  "model_groups": [
    {
      "models": {
        "model-name-1": {
          "target": ["stat-type": "stat-name"], # stat-type=stat|calc|extra
          # additional parameters just for this model
        },
        ...
      }
      # other parameters (key/value) that override and add to global 
      # defaults and pertain just to this model group
    },
    ...
  ]
}
```
4. Run the data export scripts to generate the parquet data files used for training.
1. Create the models using the cli in lib. Each model will likely output 2 files, a model definition file and a model artifact (the actual model saved as a pickle).
```python -m lib.cli```
1. (Optional) Load the models into the sport database and run some tests. Load modules using 
model_manager.py from the fantasy repository. Generate lineups or run backtesting using one
of the debug configurations or lineup.sc or backtest.sc
1. Archive the models using model_archive.py from the fantasy repository. Prior to archival make sure that mlflow is running and can be accessed. (see below)
1. Archive the training data.

## MLFLOW

### MLFlow Setup
1. [server-side] Create a new python venv, activate it and install mlflow
    ```
    python -m venv venv-mlflow
    source venv-mlflow/bin/activate
    pip install mlflow
    ```
1. [server-side] In the activated venv run the following. Be sure to set `FANTASY_MLFLOW_DIR` to the proper path.
    ```
    FANTASY_MLFLOW_DIR=wherever-this-is
    cd $FANTASY_MLFLOW_DIR
    mlflow server \
      --backend-store-uri sqlite:///mlflow.sqlite \
      --serve-artifacts \
      --artifacts-destination mlflow-artifacts \
      --gunicorn-opts "-t 120"
    ```
1. [client-side] Set up an ssh tunnel over port 5000 to the mlflow server.
1. [client-side] Go to http://localhost:5000 . Note that mlflow browser application
application stat is wonky, so if the application doesn't load either clear cookies
or run in incognito mode. If no experiments appear then the mlflow directory on the
server side is not correct or you are starting from scratch.

### Model Upload
Run the following to upload a model:
```
model_archive.py put --exp-name EXPERIMENT-NAME MODEL-FILENAME.model
```
Additional arguments that may be useful are `--active` to activate the model for use
and `--exp-desc` to add a description for the experiment

### Model Download
Run the following to download all active models to the current directory:
```
model_archive.py get --dest .
```
Use -h for other options.