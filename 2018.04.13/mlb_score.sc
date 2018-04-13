for SCORER in FDmlb DKmlb; do
    for SEASON in 2017 2016; do
        echo scoring $SCORER $SEASON
        ./scripts/calc.sc --seasons $SEASON --progress mlb.db $SCORER
    done
done
