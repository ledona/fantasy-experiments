from math import sqrt
from typing import Optional, Literal
import re

from sklearn2pmml import make_tpot_pmml_config
import autosklearn.regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict
import onnx
import onnxruntime.backend as ort_backend
import onnxruntime as ort
import pypmml

import logging


LOGGER = logging.getLogger("automl")

Frameworks = Literal['skautoml', 'tpot']

UNSUPPORTED_TPOT_MODELS = [
    'sklearn.decomposition.FastICA', 
    'tpot.builtins.ZeroCount',
    'tpot.builtins.OneHotEncoder',
    'sklearn.kernel_approximation.Nystroem',
    'sklearn.kernel_approximation.RBFSampler',
]    
    
def get_tpot_config(output_type):
    tpot_config = regressor_config_dict.copy()
    if output_type == 'pmml':
        return make_tpot_pmml_config(tpot_config)
    elif output_type == 'onnx':
        for model in UNSUPPORTED_TPOT_MODELS:
            del tpot_config[model]
        return tpot_config
    raise NotImplementedError(f"output_type '{output_type}' not supported")


def create_automl_model(
    target,
    random_state=1,
    framework: Frameworks = 'tpot',
    max_train_time=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    model_desc=None,
    target_output: None | Literal['pmml', 'onnx'] = None,
    post_process_model=None,
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
        raise NotImplementedError("skautoml export to onnx not yet supported 2022-08-25")
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
            config_dict=get_tpot_config(target_output),
            **automl_params
        )
    else:
        raise NotImplementedError(f"framework '{framework}' not supported")
    
    eval_results = None
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train, **fit_params)    
        if X_test is not None and y_test is not None:
            eval_results, predictions = error_report(model, X_test, y_test, desc=model_desc)

    if post_process_model:
        model = post_process_model(model)

    return {
        'model': model, 
        'fit_params': fit_params,
        'eval_result': eval_results,
        'predictions': predictions,
    }


def error_report(model, X_test, y_test, y_fallback: None = None,
                 desc: str = None, show_results=True) -> dict:
    """ 
    display the error report for the model, also return a dict with the scores
    
    model - an onnx, pmml or scikit-learn style model/pipeline
    """
    if isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, pypmml.model.Model):
        predictions = model.predict(X_test)
    elif isinstance(model, onnx.ModelProto):
        session = ort.InferenceSession(model.SerializeToString())
        inputs = {}
        for name in X_test.columns:
            input_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
            inputs[input_name] = np.array([[X_test[name].iloc[0]]]).astype(np.double)

        session = ort_backend.prepare(model)
        predictions = session.run([y_test.name], X_test)
        LOGGER.debug("Predictions for %s: %s", desc, predictions)
    else:
        raise NotImplementedError(f"model type {type(model)} not supported")

    if y_fallback and (nans := pd.isna(predictions[predictions.columns[0]])).any():
        nan_count = np.count_nonzero(nans)
        LOGGER.warning("Predictions for %s contain NaN. Filling %i of %i predictions with fallback value %f.",
                       desc, nan_count, len(predictions), y_fallback)
        predictions.fillna(y_fallback, inplace=True)
    r2, rmse, mae = None, None, None
    try:
        r2 = round(sklearn.metrics.r2_score(y_test, predictions), 4)
        rmse = round(sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)), 4)
        mae = round(sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)), 4)    
    except ValueError as e:
        LOGGER.error(f"Error calculating error metrics for %s: {e}", desc)
        return None, predictions
        
    result = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
    }
    if show_results:
        LOGGER.info("**** Error Report for %s ****: %s", desc, result)   

        plot_data = pd.DataFrame({
            'truth': y_test,
            'prediction': predictions
        })
        plot_data['error'] = plot_data.prediction - plot_data.truth

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

    return result, predictions


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
