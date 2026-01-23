# Model creation and management
This project folder has functionality for creating, evaluating and cataloging models. The
two types of models managed here are player/team predictive models and optimal lineup
generation models. Basic order of operations is:
1. Export data
1. Create model
1. Update the model catalog
1. If new models are good enough then import them for use

## Model Creation
Regression models are either automl generated or are deep learning models.

### Export data
Before models can be trained, training data must be constructed. To export training data:

1. Make sure that the fantasy environment is successfully installed and usable, and the database files containing raw and calculated stats are in _$FANTASY_HOME_.
1. Create a new model folder in this directory. Naming of these folders is _YYYY.MM_. Easiest is to copy the most recent model folder and rename it.
1. Export the data
    - **Create/update the data export scripts**: Review the data export scripts to make sure that the data DB name is correct, the seasons to export are correct, and all desired features are targets for all models that will be created are included. To get available positions and stats to include in the export run `dumpdata.sc DB_FILE --list`
    - **Run the data export**: Run the export script (or the part of the export script for the data required for the models being generated) to generate the parquet data files used for training.
1. Use resume flags to continue the execution of a previous data export run.
    - Some exports take forever, in which case it may be good to use the resume flags `--resume` and `--resume_dir`
    - `--inf_resume_ignore_params` can be used to ignore resume flag validation in the case where resume should not fail due to a mismatch in the flags used in a previous data export run.
    - If a previous dump needs to be resumed and a progress dir was not originally used, look in the log for the previous progress directory (saved under tmp) and either reuse that or create a new directory somewhere safe, copy the previous progress files there, then use the new directory.

### Creating Player/Team regression models
To create, archive and use new predictive models perform the following steps

1. Update the model training json files.
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
1. Create the models using the cli in lib. Each model will likely output 2 files, a model definition file and a model artifact (the actual model saved as a pickle). From the models folder execute:
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
1. (Optional) Load the models into the sport database and run some tests. Load models using 
model_manager.py from the fantasy repository (See fantasy repository's README). 
Generate lineups or run backtesting using one of the debug configurations or lineup.sc or backtest.sc .

### Deep Learning regression models
If the avg loss from one training iteration to the next is jumping around alot or not changing fast enough, increase/decrease the learning rate. As the learning rate decreases, updates to the model from one iteration to the next should lessen. Increasing the learning rate should cause models to change more from one iteration to the next.

## Model Cataloging
Run the following to create a catalog of the models in a directory and its subfolders. The catalog will be written to a csv file in the root-model-dir directory. The filename will be timestamped.
```python -m lib.regressor catalog --root {root-model-dir} --exclude active-models ".*IGNORE.*" --best```

The model catalog is where the best performing models can be identified for use as active models going forward.

## Model Performance Operations
Use the following command to repair, update, test or calculate model performance
```
python -m lib.regressor performance [MODEL_FILE_PATTERN] \
  --cfg [model_cfg.json] --data_dir [DATA_DIR] -op [repair|test|update|calc]
```

## Loading Active Models (train uncertainty predictors)
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
Note that the model config file will be retrieved from S3, not from any local (to the cloud server) source.
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