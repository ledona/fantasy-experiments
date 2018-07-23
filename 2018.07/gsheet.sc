#!/bin/bash

TYPES_mlb="OFF P"
TYPES_nfl="QB WT R K D"

usage()
{
    echo "Upload meval results to google sheets
usage: gsheet.sc mlb|nfl"
}

if [ -z "${1}" ]; then
    usage
    exit 1
fi

TYPES=TYPES_${1}

for fantasy in fd dk; do
  for type in ${!TYPES}; do
      for f in experiments/2018.07/$1*_${type}_*${fantasy}*tsv; do
          # ignore the edge case of no files
          [ -e "$file" ] || continue

          echo $f
          # subsheet names are like 'p_ols'
          subsheet=`echo $f | sed "s/.*${1}_\([^.]*\).*/\1/"`
          meval_gsheet.sc --verbose --sheet_name "models ${fantasy} ${m} 2018.07" \
                          --subsheet_title $subsheet --folder_path /fantasy/${1}/ \
                          --sort_by score_mae -- $f
    done
  done
done
