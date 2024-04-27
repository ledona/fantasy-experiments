# Model creation and management
This project folder has functionality for creating, evaluating and cataloging models. The
two types of models managed here are player/team predictive models and optimal lineup
generation models. Basic order of operations is:
1. Create model using regressor or deep-lineup cli
1. Update the model catalog
1. If new models are good enough then import them for use

## Model Creation
There are two different types of models player/team predictive models that predict
performance for individual player's and teams, and lineup models that predict
optimal daily fantasy lineups. Note that the later is not the same as optimal
lineup construction based on lineup strategy/constraints and player predictions.

### Player/Team predictive models
To create, archive and use new predictive models perform the following steps

1. Make sure that the fantasy environment is successfully installed and usable, and the 
database files containing raw and calculated stats are in _$FANTASY_HOME_.
2. Create a new model folder. Easiest to copy the most recent model folder and rename.
3. Create/update the data export scripts and model training json files. The training files are json dicts with the following structure (refer to previous files for concrete examples):
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
4. Run the data export scripts found in the model definition subfolder to generate the parquet data files used for training.
5. Create the models using the cli in lib. Each model will likely output 2 files, a model definition file and a model artifact (the actual model saved as a pickle). From the models folder execute:
```
# list models defined in a model definition file
python -m lib.regressor train {MODEL_DIR}/{SPORT}.json
# get model create params for a model
python -m lib.regressor train {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --info
# create model using defaults
python -m lib.regressor train --n_jobs 4 [--automl_type MODEL_TYPE] {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --dest {DEST_MODEL_DIR}

# for example
python -O -m lib.regressor train --n_jobs 4 --automl_type tpot 2024.02/nba.json NBA-DK \
  --data_dir /fantasy-isync/fantasy-modeling/2023.12/ --dest /fantasy-isync/fantasy-modeling/2024.04/
```
6. (Optional) Load the models into the sport database and run some tests. Load models using 
model_manager.py from the fantasy repository (See fantasy repository's README). 
Generate lineups or run backtesting using one of the debug configurations or lineup.sc or backtest.sc.

### Deep lineup models
1. Create the dataset.
1. Train the model
1. Catalog the model
1. Load the model. See fantasy repository's README for details.

### Game players/teams models
1. Review data/model configuration file
2. Create the dataset
3. Train the model
4. Catalog the model
5. Load the model

## Model Cataloging
Run the following to create a catalog of the best models currently in the catalog
```python -m lib.regressor catalog --best --root {root-model-dir}```

## Loading Active Models
TBD