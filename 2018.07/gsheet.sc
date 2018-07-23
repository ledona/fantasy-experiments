#!/bin/bash

for fantasy in fd dk; do
  for m in OFF P; do
    for f in experiments/2018.07/mlb*_${m}_*${fantasy}*tsv; do
      echo $f
      # subsheet names are like 'p_ols'
      subsheet=`echo $f | sed "s/.*mlb_\([^.]*\).*/\1/"`
      meval_gsheet.sc --verbose --sheet_name "models ${fantasy} ${m} 2018.07" \
        --subsheet_title $subsheet --folder_path /fantasy/mlb/ \
        --sort_by score_mae -- $f
    done
  done
done
