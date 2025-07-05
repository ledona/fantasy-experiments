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

### Creating Player/Team predictive models
To create, archive and use new predictive models perform the following steps

1. Make sure that the fantasy environment is successfully installed and usable, and the 
database files containing raw and calculated stats are in _$FANTASY_HOME_.
2. Create a new model folder in this directory. Naming of these folders is _YYYY.MM_. Easiest is to copy the most recent model folder and rename it.
3. Create/update the data export scripts and model training json files. 
    - Review the data export scripts to make sure that the data DB name is correct, the seasons to export are correct, and all desired features are targets for all models that will be created are included.
    - The training files are json dicts with the following structure. 
      - The union of _train_params_ are used to train a model. Values set at a lower level, more specific to the model, take precidence.
      - _train_params_ can be designated as algorithm specific by prefixing the parameter name with "_algorithm._". E.g. a parameter named _param_ that only applies to the _tpot_ algorithm should be named _tpot.param_.Algorithm specific parameters take precidence over none algorithm specific parameters of the same name.
      - The union of _train_params_ and _cols_to_drop_ are used to train a model. For _train_params_ values from the lowest (most specific to the model) level override parameters with the same name at a higher level.
      - Refer to previous files for examples.
```
{
  "global_defaults": {
    # dict with default training parameters for all models
    # defined in this configuration file
    "train_params": { ... },
    "cols_to_drop": [ ... ]
  },
  "model_groups": [
    {
      "train_params": { ... },
      "cols_to_drop": [ ... ]
      "models": {
        "model-name-1": {
          "target": ["stat-type": "stat-name"], # stat-type=stat|calc|extra
          # additional parameters just for this model
          "train_params": { ... },
          "cols_to_drop": [ ... ]
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

# view model create params for a model
python -m lib.regressor train --data_dir {PATH_TO_DIR_W_DATA_FILES} {MODEL_DIR}/{SPORT}.json {MODEL_NAME} --info

# create model using defaults
python -m lib.regressor train --data_dir {PATH_TO_DIR_W_DATA_FILES} --dest_dir {DEST_MODEL_DIR} \
  [--algo MODEL_TYPE] [--slack] {MODEL_DIR}/{SPORT}.json {MODEL_NAME}

# create multiple models
python -m lib.regressor train --data_dir {PATH_TO_DIR_W_DATA_FILES} --dest_dir {DEST_MODEL_DIR} \
  [--algo MODEL_TYPE] [--slack] {MODEL_DIR}/{SPORT}.json \
  ({MODEL_NAME_W_WILDCARDS} | [--models {MODEL_NAME} {MODEL_NAME} ...])

# create a model based on an existing model 
python -m lib.regressor retrain --data_dir {PATH_TO_DIR_W_DATA_FILES} --dest_dir {DEST_MODEL_DIR} \
  [--orig_cfg_file {MODEL_DIR}/{SPORT}.json] [--slack] {EXISTING_MODEL_FILEPATH}

# example which forces tpot and specifies number of processes
python -O -m lib.regressor train --n_jobs 4 --algo tpot 2024.02/nba.json NBA-DK \
  --data_dir /fantasy-isync/fantasy-modeling/2024.04/pt \
  --dest_dir /fantasy-isync/fantasy-modeling/2024.04/data

# example that uses a simple NN
python -m lib.regressor train 2024.12/nfl.json "NFL-QB-
DK" \
  --data_dir /fantasy-isync/fantasy-modeling/2024.12/data/ \
  --dest_dir /fantasy-isync/fantasy-modeling/2024.12/pt \
  --algo nn --lr .00001 --layers 3 --max_epochs 500 --early 10
```
6. (Optional) Load the models into the sport database and run some tests. Load models using 
model_manager.py from the fantasy repository (See fantasy repository's README). 
Generate lineups or run backtesting using one of the debug configurations or lineup.sc or backtest.sc.

### NN models
- If the avg loss from one training iteration to the next is jumping around alot or not changing fast enough, increase/decrease the learning rate. As the learning rate decreases, updates to the model from one iteration to the next should lessen. Increasing the learning rate should cause models to change more from one iteration to the next.

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
Run the following to create a catalog of the models in a directory and its subfolders. The catalog will be written to a csv file in the root-model-dir directory. The filename will be timestamped.
```python -m lib.regressor catalog --root {root-model-dir} --exclude active-models ".*IGNORE.*" --best```

## Model Performance Operations
Use the following command to repair, update, test or calculate model performance
```
python -m lib.regressor performance [MODEL_FILE_PATTERN] \
  --cfg [model_cfg.json] --data_dir [DATA_DIR] -op [repair|test|update|calc]
```

## Loading Active Models
The active model folder is defined in the environment variable __FANTASY_MODELS_PATH__. Uncertainty/error estimation models are also required for numerous things and can be trained at this step.
```
# model_manager.py import {paths to .model files or a single .csv file with best models}
# model_manager.py fit-uncertainty --data_dir {path to data used to train the original models} {models to train, glob patterns supported}
```

## Cloud Training
1. Set up for training by following the instructions in the fantasy repo configuration management folder
2. To train a model use the _aws_train.sc_ script in this folder. For example:
```
# ./aws_train.sc {S3-BUCKET} {DEST-DIR} {MODEL-CFG-FILE} {MODEL-NAME} {training args ...}
# for example
cd /fantasy-experiments/models
./aws_train.sc s3://ledona-fantasy /tmp/models mlb.json "MLB-H-*" --exists reuse --algo tpot --slack --n_jobs 4
```
3. To copy/sync model results from S3 use aws cli.
```
# install aws cli
sudo apt-get install awscli
# configure/setup security
aws configure

# copy
aws s3 cp {S3-models-path} {local-models-path} [--exclude "*" --include "MLB*"] [--dryrun]
# or sync, this will only copy things in s3 that are not in/don't match the destination
aws s3 sync {S3-models-path} {local-models-path} [--dryrun]
```