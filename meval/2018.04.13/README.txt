To generate NBA gsheets run

SHEET_PRE="2018.04.13"
for SCORER in dk fd; do
    echo working on $SCORER
    for F in $FANTASY_HOME/experiments/2018.04.13/nba_fantasy*$SCORER*.tsv; do
        echo $F
        SUBSHEET_TITLE=`echo $F | sed "s/.*nba_fantasy_\(.*\)\.$SCORER.*/\1/"`
        SHEET_NAME="$SHEET_PRE $SCORER model"
        echo sheet=\"$SHEET_NAME\" subsheet=\"$SUBSHEET_TITLE\"
        echo meval_gsheet.sc --verbose --sheet_name "$SHEET_NAME" --subsheet_title "$SUBSHEET_TITLE" \
                                        --folder_path "fantasy/nba" "$F"
    done
    echo done with $SCORER
    echo
done


MLB gsheets

SHEET_PRE="2018.04.13"
for SCORER in dk fd; do
    for MTYPE in p off; do
        echo working on scorer=$SCORER mtype=$MTYPE
        for F in $FANTASY_HOME/experiments/2018.04.13/mlb_fantasy_${MTYPE}_*\.$SCORER*.tsv; do
            echo FILE=$F
            SUBSHEET_TITLE=`echo $F | sed "s/.*mlb_fantasy_${MTYPE}_\(.*\)\.${SCORER}.*tsv/\1/"`
            if [ "$SUBSHEET_TITLE" == "$F" ]; then
               echo error figuring out subsheet title for $F
               break 3
            fi
            SHEET_NAME="$SHEET_PRE $SCORER $MTYPE model"
            echo sheet=\"$SHEET_NAME\" subsheet=\"$SUBSHEET_TITLE\"
            meval_gsheet.sc --verbose --sheet_name "$SHEET_NAME" --subsheet_title "$SUBSHEET_TITLE" \
                            --folder_path "fantasy/mlb" "$F"
        done
        echo done with $SCORER $MTYPE
        echo
    done
done
