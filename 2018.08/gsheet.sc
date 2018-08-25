#!/bin/bash

TYPES_mlb="OFF P"
TYPES_nfl="QB WT R K D"
TYPES_nba="."

FANTASIES="fd dk y"

usage()
{
    echo "Upload meval results to google sheets
usage: gsheet.sc mlb|nfl|nba"
}

if [ "$1" != "mlb" -a "$1" != "nfl" -a "$1" != "nba" ]; then
    usage
    exit 1
fi

TYPES=TYPES_${1}

# need globbing, and some of the other scripts disable so...
set +f

for fantasy in $FANTASIES; do
    for TYPE in ${!TYPES}; do
        echo $fantasy $TYPE
        if [ "${TYPE}" == "." ]; then
            TYPE_=""
        else
            TYPE_="${TYPE}_"
        fi
        for f in experiments/2018.08/$1*_${TYPE_}*${fantasy}*tsv; do
            # ignore the edge case of no files
            [ -e "$f" ] || continue

            echo
            echo $f
            # subsheet names are like 'p_ols'
            subsheet=`echo $f | sed "s/.*${1}_\([^.]*\).*/\1/"`
            python scripts/meval_gsheet.sc --verbose \
                   --sheet_name "models ${fantasy} 2018.08" \
                   --subsheet_title $subsheet --folder_path /fantasy/${1}/ \
                   --sort_by score_mae -- $f
        done
    done
done
