from math import sqrt
from typing import Optional, Literal

import warnings
warnings.filterwarnings("ignore", "Warning: optional dependency `torch` is not available. - skipping import of NN models.")

import autosklearn.regression
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from tpot import TPOTRegressor

import logging


LOGGER = logging.getLogger("automl")

Frameworks = Literal['skautoml', 'tpot']


def create_automl_model(
        target,
        pca_components=None,
        random_state=1,
        framework: Frameworks = 'skautoml',
        max_train_time=None,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model_desc=None,
        **automl_params
):
    """
    create the model

    pca_components - if not None then add a PCA transformation step
       prior to model fit
    max_train_time - time to train the model in seconds
    X_train, y_train - if not None then train the model
    X_test, y_test - if not None then score
    model_desc - required if X_test and y_test are not None
    **automl_params - used when creating the model object

    returns - dict containing model, fit_params and evaluation results
    """
    fit_params = {}
    if framework == 'skautoml':
        if max_train_time is None:
            raise ValueError("max_train_time must not be None for skautoml")
        model = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=max_train_time,
            seed=random_state,
            **automl_params
        )
        if pca_components is not None:
            fit_params['automl__dataset_name'] = target
        else:
            fit_params['dataset_name'] = target

    elif framework == 'tpot':
        if max_train_time is not None:
            if (rem:= max_train_time % 60) != 0:
                new_train_time = max_train_time + 60 - rem
                LOGGER.warning(f"TPot requires a training time in minutes. Rounding requested train time from {max_train_time} up to {new_train_time} seconds")
                max_train_time = new_train_time
            max_train_time /= 60
        model = TPOTRegressor(
            random_state=random_state,
            max_time_mins=max_train_time,
            **automl_params
        )
    else:
        raise NotImplementedError(f"framework '{framework}' not supported")

    if pca_components is not None:
        model = Pipeline([
            ('pca', PCA(n_components=pca_components, random_state=random_state)),
            ('automl', model),
        ])

    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train, **fit_params)
    if X_test is not None and y_test is not None:
        results = error_report(model, X_test, y_test, desc=model_desc)

    return {
        'model': model, 
        'fit_params': fit_params,
        'eval_result': results
    }


def error_report(model, X_test, y_test, desc: str = None) -> dict:
    """ 
    display the error report for the model, also return a dict with the scores
    """
    if desc:
        print(desc)

    predictions = model.predict(X_test)
    print(
        "R2 score:",
        (r2 := round(sklearn.metrics.r2_score(y_test, predictions), 4))
    )
    print(
        "RMSE score:",
        (rmse := round(sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)), 4))
    )
    print(
        "MAE score:",
        (mae := round(sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)), 4))
    )

    plot_data = pd.DataFrame({
        'truth': y_test,
        'prediction': predictions
    })
    plot_data['error'] = plot_data.prediction - plot_data.truth
    # display(plot_data)

    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle(f"{desc or 'unknown model'} : {r2=} {rmse=} {mae=}")
    for ax in axs:
        ax.axis('equal')

    min_v = min(plot_data.truth.min(), plot_data.prediction.min())
    max_v = max(plot_data.truth.max(), plot_data.prediction.max())

    axs[0].plot((min_v, max_v),
                (min_v, max_v),
                '-g', linewidth=1)
    plot_data.plot(kind='scatter', x='truth', y='prediction', ax=axs[0])

    axs[1].yaxis.set_label_position("right")
    axs[1].plot((min_v, max_v),
                (0, 0),
                '-g', linewidth=1)
    plot_data.plot(kind='scatter', x='truth', y='error', ax=axs[1])
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
    }


def get_df_types(df, drop: Optional[list[str]] = None) -> list:
    """
    identify the input types for the model, based on function found at
    https://docs.microsoft.com/en-us/azure/azure-sql-edge/deploy-onnx
    """
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float32':
            t = FloatTensorType([None, 1])
        elif v == 'float64':
            t = DoubleTensorType([None, 1])
        else:
            raise ValueError(f"Unsupported type {v}")
        inputs.append((k, t))
    return inputs
