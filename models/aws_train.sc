#!/bin/bash


# Exit on any error
set -e

pushd /fantasy-experiments/models

function usage() {
    echo "AWS Model Trainer
Usage: $0 [--dryrun] AWS-S3-PATH LOCAL-DEST-DIR MODEL-FILE MODEL-NAME [ARG1 ARG2 ...]

This script is for training models on an AWS instance. Training is accomplished via the following steps:
1) create a temp destination folder for models at the at the path in the cli arg LOCAL-DEST-DIR
2) copy model definition file for the cli arg MODEL-FILE from S3 if it isn't already available 
   locally from a previous run of this script.
3) run training in --info mode to get the name of the data file
4) copy the data file from s3 if it is not available locally from a previous run of this script.
5) Train model(s) matching the cli arg MODEL-NAME. MODEL-NAME can contain wildcards. Wildcard character is
   '*'. All command line args for this script after the MODEL-FILE are passed to the training program.
"
    exit 1
}

DRYRUN=false

if [ "$1" = "--dryrun" ]; then
    DRYRUN=true
    echo "DRYRUN enabled!
"
    shift 1
fi

S3_PATH=$1
DEST_DIR=$2
MODEL_FILE=$3
MODEL_NAME=$4

if [ -z "$S3_PATH" ] || [ -z "$MODEL_FILE" ] || [ -z "$DEST_DIR" ] || [ -z "$MODEL_NAME" ]; then
    usage
fi

shift 4  # Remove parsed arguments, leaving only optional args

# Copy MODEL_FILE from S3 if not present locally
cmd="aws s3 cp '${S3_PATH}/${MODEL_FILE}' ."
if [ $DRYRUN = true ]; then
    echo "# if the model file does not exist copy it from '${S3_PATH}'"
    echo $cmd
elif [ ! -f "$MODEL_FILE" ]; then
    echo "# Model file '$MODEL_FILE' not found, copying from '${S3_PATH}'"
    eval "$cmd"
else
    echo "# Model file '$MODEL_FILE' found locally"
fi

train_cmd="python -m lib.regressor train $MODEL_FILE $MODEL_NAME --dest_dir $DEST_DIR $@"

# Run train in info mode and capture output
cmd="$train_cmd --info"
echo "# run train in info mode to get data filename"
echo $cmd
if [ $DRYRUN = false ]; then
    INFO=$($cmd)
    echo 
    echo ------------------ info output --------------------
    echo $INFO
    echo ---------------------------------------------------
    echo
    # Extract DATAFILE filename
    DATAFILE=$(echo "$INFO" | grep "'data_filename'" | head -n 1 | sed "s/.*'data_filename': '\([^']*\)'.*/\1/")
    if [ -z "$DATAFILE" ]; then
        echo "Error: Could not extract filename from INFO"
        exit 1
    fi
    echo "Data filename is '$DATAFILE'"
else
    DATAFILE=DATAFILE.parquet
fi

# Check if file not found message exists in INFO
cmd="aws s3 cp '${S3_PATH}/${DATAFILE}' ."
if [ $DRYRUN = true ]; then
    echo "# if data file is not local copy it from S3"
    echo $cmd
elif echo "$INFO" | grep -q "Data file '${DATAFILE}' was not found"; then
    echo "Data file '${DATAFILE}' was not found, copying from S3"
    echo $cmd
    eval "$cmd"
else
    echo "Data file '${DATAFILE}' found locally"
fi

# Run training with remaining args and temp directory
echo
echo "# ---------- Run training ----------"
echo $train_cmd
if [ $DRYRUN = false ]; then
    eval "$train_cmd"
fi

echo "# --------- Training of ${MODEL_NAME} FINISHED! --------"
popd