To create, archive and use new predictive models perform the following steps

0. Make sure that the fantasy environment is successfully installed and usable, and the 
database files containing raw and calculated stats are in $FANTASY_HOME.
1. Create a new model folder. Easiest to copy the most recent model folder and rename.
2. Update the contents (notebooks) in the folder to reflect the updated data, stats and
code required to extract data and train/evaluate the new models.
3. Run the inference data export program (as defined in the notebook) to export the training
and evaluation data to parquet and/or csv files.
4. Run the notebooks to create the new models. Each model will likely output 2 files,
a pickle file with the model artifact and a .model file with json that describes the model.
5. (Optional) Load the models into the sport database and run some tests. Load modules using 
model_manager.py from the fantasy repository. Generate lineups or run backtesting using one
of the debug configurations or lineup.sc or backtest.sc
6. Archive the models using model_archive.py from the fantasy repository. Prior to archival 
make sure that mlflow is running and can be accessed
7. Archive the training data.