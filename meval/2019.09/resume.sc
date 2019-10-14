#!/bin/bash

RESUME=true
script_dir="$(dirname "$0")"
source ${script_dir}/env.sc

echo python -O ${FANTASY_HOME}/scripts/meval_resume.sc $CACHE_ARGS --slack $1
