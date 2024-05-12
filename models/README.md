# Model creation and management
This project folder has functionality for creating, evaluating and cataloging models. The
two types of models managed here are player/team predictive models and optimal lineup
generation models. Basic order of operations is:
1. Create model using regressor or deep-lineup cli
1. Update the model catalog
1. If new models are good enough then import them for use

## Model Creation
There are three different types of models
1. player/team predictive models that predict performance for individual players and teams
1. lineup models that predict optimal daily fantasy lineups
1. full game models that predict all player and team stats for a game in one shot, as opposed to predicting each player/team stat individuals

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
python -m lib.regressor train --n_jobs 4 [--algo MODEL_TYPE] {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --dest {DEST_MODEL_DIR}

# for example
python -O -m lib.regressor train --n_jobs 4 --algo tpot 2024.02/nba.json NBA-DK \
  --data_dir /fantasy-isync/fantasy-modeling/2024.04/pt \
  --dest_dir /fantasy-isync/fantasy-modeling/2024.04/data
```
6. (Optional) Load the models into the sport database and run some tests. Load models using 
model_manager.py from the fantasy repository (See fantasy repository's README). 
Generate lineups or run backtesting using one of the debug configurations or lineup.sc or backtest.sc.

### Deep lineup models
Deep lineup models take as input a slate of games, including all player costs, and predicted scoring. The model output is a lineup to bet. As an intermediate step a DNN is used to take the input and infer player weights which are used along with cost information to create an optimized lineup. To create/use models:

1. Create the dataset - Each sample in a dataset is a randomly generated slate of games along with starting players, predicted fantasy scores, historic fantasy scores, slate cost and other data describing the slate. Creation of a dataset is done by first creating a player/team predictive model dataset, then using ```python -m lib.deep_lineup data``` to create the deep lineup dataset. 
1. Train the model - Use ```python -m lib.deep_lineup train``` to train a model.
1. Catalog the model (see below)
1. Load the model. See fantasy repository's README for details on using ```model_manager```
1. To generate a lineup use ```lineup.sc --deep```

### Full Game players/teams models
These models predict outcomes for all players/teams involved in a game. The sample input is historic stats for all players and teams in a game.

1. Review data/model configuration file
1. Create the dataset - ```python -m lib.all_game_pt [CFG-FILE] [MODEL-NAME] data [INPUT-DATA-DIR] [TRAINING-DATASET-FILEPATH]```
1. Train the model - ```python -m lib.all_game_pt [CFG-FILE] [MODEL-NAME] train [TRAINING-DATASET-FILEPATH]``` where _TRAINING-DATASET-FILEPATH_ is folder used when creating the dataset
1. Catalog the model (see below)
1. Load the model. See fantasy repository's README for details on using ```model_manager```
1. To generate a lineup use ```lineup.sc --game_model

## Model Cataloging
Run the following to create a catalog of the best models currently in the catalog
```python -m lib.regressor catalog --best --root {root-model-dir}```

## Loading Active Models
TBD