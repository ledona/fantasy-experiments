for SCORER in FDnba DKnba; do
    for SEASON in 2016 2015 2014 2013 2012; do
        echo scoring $SCORER $SEASON
        ./scripts/calc.sc --progress --season 2017 nba.db $SCORER
    done
done
