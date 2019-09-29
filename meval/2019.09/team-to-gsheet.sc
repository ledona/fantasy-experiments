# copy team score prediction tsv files to gsheets

SPORT=mlb

for MODEL in OLS BLE XG DNN_ADA DNN_RS; do
    for tsv_file in ${SPORT}_team*${MODEL}*tsv; do
        echo processing: $tsv_file
        if [ ! -f "$tsv_file" ]; then
            echo nothing found for \"./"$tsv_file"\"
            continue
        fi

        cmd="gsheet.sc --verbose --sheet_name ${SPORT}_team_score --folder_path '/fantasy/mlb' \
                  --subsheet_title $MODEL \
                  --sort_by score_mae -- $tsv_file"
        echo $cmd
        eval $cmd
    done
done
