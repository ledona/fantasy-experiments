#!/bin/bash


# Exit on any error
set -e

function usage() {
    if [ -n "$1" ]; then
        echo "Error: $1"
        echo
    fi
    echo "Cloud Model Trainer
Usage: $0 [--dryrun] [--s3-profile PROFILE] [--s3-path S3_PATH] [--local-dest-dir DEST_DIR] [--exclude-remote-models] MODEL-FILE MODEL-NAME [ARG1 ARG2 ...]

--s3-path: S3 path to use. If not specified, falls back to the S3_BUCKET env variable.
--s3-profile: AWS profile to use (default: 'default'). Can also be set via S3_PROFILE env variable.
--local-dest-dir: Local destination directory for models (default: /tmp/models).
--exclude-remote-models: List *.model files in \$S3_PATH/models and write their names to \$DEST_DIR/exclude-from-training.txt before training.
S3_ENDPOINT_URL env variable can optionally be set to specify a custom S3 endpoint (e.g. Cloudflare R2).
On AWS, S3 access should be granted through AWS permissions to the VPC, which means
access should already be granted. If Cloudflare R2 is being used then ~/.aws/credentials
is required and should have been copied during the setup script run.

Training proceeds using the following steps:
1) copy model definition file for the cli arg MODEL-FILE from S3 if it isn't already available
   locally from a previous run of this script.
2) optionally write names of existing remote *.model files to exclude-from-training.txt (--exclude-remote-models)
3) run training in --info mode to get the name of the data file
4) copy the data file from s3 if it is not available locally from a previous run
5) Train model(s) matching the cli arg MODEL-NAME. MODEL-NAME can contain wildcards. Wildcard character is
   '*'. All command line args for this script after the MODEL-FILE are passed to the training program.
"
    exit 1
}

DRYRUN=false
S3_PROFILE=${S3_PROFILE:-default}
S3_PATH=""
DEST_DIR=/tmp/models
EXCLUDE_REMOTE_MODELS=false

S3_ENDPOINT_PARAM=""
if [ -n "$S3_ENDPOINT_URL" ]; then
    S3_ENDPOINT_PARAM="--endpoint-url $S3_ENDPOINT_URL"
fi

while true; do
    if [ "$1" = "--dryrun" ]; then
        DRYRUN=true
        echo "DRYRUN enabled!
"
        shift 1
    elif [ "$1" = "--s3-profile" ]; then
        if [ -z "$2" ]; then
            usage "--s3-profile requires an argument"
        fi
        S3_PROFILE=$2
        shift 2
    elif [ "$1" = "--s3-path" ]; then
        if [ -z "$2" ]; then
            usage "--s3-path requires an argument"
        fi
        S3_PATH=$2
        shift 2
    elif [ "$1" = "--local-dest-dir" ]; then
        if [ -z "$2" ]; then
            usage "--local-dest-dir requires an argument"
        fi
        DEST_DIR=$2
        shift 2
    elif [ "$1" = "--exclude-remote-models" ]; then
        EXCLUDE_REMOTE_MODELS=true
        shift 1
    elif [[ "$1" == --* ]]; then
        usage "unknown option: $1"
    else
        # break when we hit the first positional arg, this should be the model cfg file
        break
    fi
done

if [ -z "$S3_PATH" ]; then
    if [ -z "$S3_BUCKET" ]; then
        usage "--s3-path must be provided or S3_BUCKET environment variable must be set"
    fi
    S3_PATH=$S3_BUCKET
fi

MODEL_FILE=$1
MODEL_NAME=$2

if [ -z "$MODEL_FILE" ]; then
    usage "MODEL-FILE is required"
fi
if [ -z "$MODEL_NAME" ]; then
    usage "MODEL-NAME is required"
fi

shift 2  # Remove parsed arguments, leaving only optional args

echo "----------------------------------------"
echo "  S3 path:      ${S3_PATH}"
echo "  S3 profile:   ${S3_PROFILE}"
echo "  Endpoint URL: ${S3_ENDPOINT_URL:-<none>}"
echo "  Dest dir:     ${DEST_DIR}"
echo "  Model file:   ${MODEL_FILE}"
echo "  Model name:   ${MODEL_NAME}"
echo "  Excl. models: ${EXCLUDE_REMOTE_MODELS}"
echo "----------------------------------------"
echo

# Copy MODEL_FILE from S3 if not present locally
cmd="aws --profile ${S3_PROFILE} ${S3_ENDPOINT_PARAM} s3 cp '${S3_PATH}/${MODEL_FILE}' ."
if [ $DRYRUN = true ]; then
    echo "# if the model file does not exist copy it from '${S3_PATH}'"
    echo $cmd
elif [ ! -f "$MODEL_FILE" ]; then
    echo "# Model file '$MODEL_FILE' not found, copying from '${S3_PATH}'"
    eval "$cmd"
else
    echo "# Model file '$MODEL_FILE' found locally"
fi

# List remote *.model files and write names to exclude-from-training.txt
if [ $EXCLUDE_REMOTE_MODELS = true ]; then
    if [ ! -d "$DEST_DIR" ]; then
        echo "Error: DEST_DIR '${DEST_DIR}' does not exist"
        exit 1
    fi
    ls_cmd="aws --profile ${S3_PROFILE} ${S3_ENDPOINT_PARAM} s3 ls '${S3_PATH}/models/'"
    if [ $DRYRUN = true ]; then
        echo "# list remote models to build exclude-from-training.txt:"
        echo $ls_cmd
    else
        eval "$ls_cmd" | awk '{print $NF}' | grep '\.model$' > "${DEST_DIR}/exclude-from-training.txt"
        echo "# wrote $(wc -l < "${DEST_DIR}/exclude-from-training.txt") model names to ${DEST_DIR}/exclude-from-training.txt"
    fi
fi

train_cmd="python -m lib.regressor train $MODEL_FILE $MODEL_NAME --dest_dir $DEST_DIR $@"
if [ $EXCLUDE_REMOTE_MODELS = true ]; then
    train_cmd="$train_cmd --exclude_model_file ${DEST_DIR}/exclude-from-training.txt"
fi

# Run train in info mode and capture output
cmd="$train_cmd --info"
echo "# run train in info mode to get data filename"
echo $cmd
if [ $DRYRUN = false ]; then
    INFO=$($cmd)
    echo 
    echo ------------------ info output --------------------
    echo "$INFO"
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
cmd="aws --profile ${S3_PROFILE} ${S3_ENDPOINT_PARAM} s3 cp '${S3_PATH}/${DATAFILE}' ."
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
