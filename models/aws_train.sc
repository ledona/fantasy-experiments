#!/bin/bash

# Exit on any error
set -e

function usage() {
    echo "AWS Model Trainer
Usage: $0 [--dryrun] AWS-S3-PATH MODEL-FILE [ARG1 ARG2 ...]

This script is for training models on an AWS instance. This is done by:
1) creating a temp destination folder for models
2) copying the model definition file from S3
3) running training in --info mode to get the name of the data file
4) copying the data file (if necessary) from s3
5) training the model, all command line args after the MODEL-FILE are passed to the training program
6) uploading the trained model back to s3
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
MODEL_FILE=$2

if [ -z "$S3_PATH" ] || [ -z "$MODEL_FILE" ]; then
    usage
fi

shift 2  # Remove first two arguments, leaving only optional args

# Create temp directory
cmd="mktemp -d"
if [ $DRYRUN = false ]; then
    DEST_DIR=$($cmd)
    trap 'rm -rf "$DEST_DIR"' EXIT
    echo "Temp destination directory set to '$DEST_DIR'"
else
    echo "# create tmpdir"
    echo $cmd
    DEST_DIR=DEST-DIR
fi

# Copy MODEL_FILE from S3 if not present locally
cmd="aws s3 cp '${S3_PATH}/${MODEL_FILE}' ."
if [ $DRYRUN = true ]; then
    echo "# if the model file does not exist copy it from '${S3_PATH}'"
    echo $cmd
elif [ ! -f "$MODEL_FILE" ]; then
    echo "# Model file '$MODEL_FILE' not found, copying from '${S3_PATH}'"
    eval "$cmd"
fi

train_cmd="python -m lib.regressor train --dest_dir '$DEST_DIR' $@"

# Run regressor in info mode and capture output
cmd="$train_cmd --info"
echo "# run train in info mode to get data filename"
echo $cmd
if [ $DRYRUN = false ]; then
    INFO=$($cmd)
    echo ------------------ info output --------------------
    echo $INFO
    echo ---------------------------------------------------
    # Extract DATAFILE filename
    DATAFILE=$(echo "$INFO" | grep "'data_filename'" | sed "s/.*'data_filename': '\([^']*\)'.*/\1/")
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
echo "# Run training"
echo $train_cmd
if [ $DRYRUN = false ]; then
    eval "$cmd"
fi

if [ $DRYRUN = false ]; then
    # Check if required files exist in temp directory
    if ! ls "$DEST_DIR"/* >/dev/null 2>&1; then
        echo "Error: training failed to create output files"
        exit 1
    fi
    MODEL_FILES="$DEST_DIR"/*
else
    MODEL_FILES="MODEL-DEF.model MODEL-ARTIFACT.pkl"
fi

echo "# Copy model files from temp directory to S3"
for file in $MODEL_FILES; do
    cmd="aws s3 cp '$file' '${S3_PATH}/'"
    echo $cmd
    if [ $DRYRUN = false ]; then
        eval $cmd
    fi
done