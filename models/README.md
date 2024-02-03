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
1. Run the data export scripts to generate the parquet data files used for training.
1. Create the models using the cli in lib. Each model will likely output 2 files, a model definition file and a model artifact (the actual model saved as a pickle). From the models folder execute:
```
# list models defined in a model definition file
python -m lib.cli train {MODEL_DIR}/{SPORT}.json
# get model create params for a model
python -m lib.cli train {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --info
# create model using defaults
python -m lib.cli train --tpot_jobs 3 [--automl_type MODEL_TYPE] {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --dest {MODEL_DIR}
```
1. (Optional) Load the models into the sport database and run some tests. Load modules using 
model_manager.py from the fantasy repository. Generate lineups or run backtesting using one
of the debug configurations or lineup.sc or backtest.sc
1. Archive models and data

## Model Cataloging

## Loading Active Models