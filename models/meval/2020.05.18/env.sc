# shared environment variables for meval analyses
# sets get_meval_base_cmd, and CALC_... variables
# If FOLDS is set in the environment then it is used as the number of cross folds
#   if not 3fold is used


CACHE_ARGS="--cache_dir ./cache_dir"

# for this experiment will be used djshadow as the remote data repository
if [[ $HOSTNAME != babylon5* ]]; then
    CACHE_ARGS="${CACHE_ARGS} --cache_remote babylon5:working/fantasy/cache_dir"
fi

if [[ $RESUME == true ]]; then
    echo python -O ${FANTASY_HOME}/scripts/meval_resume.sc $CACHE_ARGS --ignore_version --slack --progress $1
    exit 0
fi

if [ "$MAX_OLS_FEATURES" == "" ]; then
    echo Error! MAX_FEATURES must be set before calling env.sc
    exit 1
fi

if [ "$MAX_CASES" == "" ]; then
    echo Error! MAX_CASES must be set before calling env.sc
    exit 1
fi

if [ "$SEASONS" == "" ]; then
    echo Error! SEASONS must be set before calling env.sc
    exit 1
fi

_SHARED_MEVAL_ARGS="${CACHE_ARGS}
                    --progress --search_bayes_scorer mae
                    --scoring mae r2"

if [ "$FOLDS" == "" ]; then
   FOLDS=3
fi

# full meval args
MEVAL_ARGS="${_SHARED_MEVAL_ARGS} --slack
            --seasons ${SEASONS}
            --search_method bayes --search_iters 70
            --search_bayes_init_pts 7
            --folds $FOLDS"

COMMON_CALC_ARGS="--n_games_range 1 7
                  --n_cases_range 500 $MAX_CASES"

# limited data for testing
_RECENT_SEASON=`echo $SEASONS | cut -d " " -f1`
TEST_MEVAL_ARGS="${_SHARED_MEVAL_ARGS}
            --seasons ${_RECENT_SEASON}
            --search_method bayes --search_iters 7
            --search_bayes_init_pts 3
            --folds 2"
TEST_COMMON_CALC_ARGS="--n_games_range 1 3
                       --n_cases_range 10 50"


CALC_OLS="sklearn --est ols
        --hist_agg_list mean median
        --n_features_range 1 ${MAX_OLS_FEATURES}"

CALC_BLE='sklearn
        --hist_agg_list mean median none
        --alpha_range .00001 1  --alpha_range_def 6 log
        --l1_ratio_range .05 .95
        --est_list br lasso elasticnet'

CALC_RF='sklearn
       --hist_agg_list mean median none
       --rf_trees_range 5 25 --rf_max_features_list sqrt log2
       --rf_min_samples_leaf_range 1 200
       --rf_crit_list mse mae --rf_max_depth_list 0 500
       --rf_n_jobs 3
       --est rforest'

_SHARED_DNN='keras
           --hist_agg_list mean median none
           --normalize
           --steps_range 100 1000 --steps_range_inc 100
           --layers_range 1 5
           --units_range 20 100
           --activation_list linear relu tanh sigmoid
           --dropout_range .3 .7'

CALC_DNN_RS="$_SHARED_DNN
           --learning_method_list rmsprop sgd
           --lr_range .005 .01"

CALC_DNN_ADA="$_SHARED_DNN
            --learning_method_list adagrad adadelta adam adamax nadam"

CALC_XG="xgboost
       --hist_agg_list mean median none
       --learning_rate_range .01 .2
       --subsample_range .5 1
       --min_child_weight_range 1 10
       --max_depth_range 3 10
       --gamma_range 0 10000 --gamma_range_def 10 log
       --colsample_bytree_range 0.5 1
       --rounds_range 75 150"

CALC_GP='sklearn
       --hist_agg_list mean median none
       --gp_kernel_list rbf rq dot mat sine
       --gp_add_white_list true false
       --est gauss'


# validate $MODEL
if [ "$MODEL" == "" ]; then
    echo Error! MODEL must be set before calling env.sc
    exit 1
else
    CALC_ARGS_NAME=CALC_${MODEL}
    if [ "${!CALC_ARGS_NAME}" == "" ]; then
        echo "Error! '${MODEL}' is not a valid model name."
        exit 1
    fi
fi


# shared args that go at the end of the calc args
CALC_POST_ARGS=""



# echo the start of the meval python command
# takes 1 arg, a string of either "--test" or an empty string
# if it returns 1 then the caller should exit (due to arg parse failure)
get_meval_base_cmd()
{
    TEST_ARG=$1

    if [ "$TEST_ARG" == '--test' ]; then
        ARGS=$TEST_MEVAL_ARGS
        RUNNER="meval.sc"
    elif [ "$TEST_ARG" == "" ]; then
        ARGS=$MEVAL_ARGS
        RUNNER="python -O ${FANTASY_HOME}/scripts/meval.sc"
    else
        exit 1
    fi

    echo "$RUNNER $ARGS"
}

# return the calc args
# takes 2 parameters, first is the calc nae, second is either --test or an empty string
# if it returns 1 then the caller should exit (due to arg parse failure)
get_calc_args()
{
    CALC_ARGS_NAME=CALC_${1}
    CALC_ARGS=${!CALC_ARGS_NAME}

    if [ -z "$CALC_ARGS" ]; then
        # unknown model
        exit 1
    fi

    TEST_ARG="$2"

    if [ "$TEST_ARG" == "--test" ]; then
        CALC_ARGS="${CALC_ARGS} ${TEST_COMMON_CALC_ARGS}"
    elif [ "$TEST_ARG" == "" ]; then
        CALC_ARGS="${CALC_ARGS} ${COMMON_CALC_ARGS}"
    else
        exit 1
    fi

    echo $CALC_ARGS $CALC_POST_ARGS
}
