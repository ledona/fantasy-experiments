import os
import shutil
from math import sqrt
from typing import Optional

import autosklearn.regression
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType


def automl(model_name,
           train_time=60, per_run_time_limit=10,
           pca_components=None,
           overwrite: bool = False,
           seed=1) -> tuple:
    """
    create the model

    overwrite - overwrite the output folder
    pca_components - if not None then add a PCA transformation step
       prior to model fit

    returns - (model, fit_params)
    """
    output_folder = '/tmp/autosklearn_regression_' + model_name
    if overwrite and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    model = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=train_time,
        per_run_time_limit=per_run_time_limit,
        output_folder=output_folder,
        seed=seed,
    )

    if pca_components is not None:
        model = Pipeline([
            ('pca', PCA(n_components=pca_components)),
            ('automl', model),
        ])
        fit_params = {'automl__dataset_name': model_name}
    else:
        fit_params = {'dataset_name': model_name}

    return model, fit_params


def error_report(model, X_test, y_test, desc: str):
    print(desc)
    # print(model.show_models())
    predictions = model.predict(X_test)
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    print("RMSE score:", sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)))
    print("MAE score:", sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)))

    plot_data = pd.DataFrame({
        'truth': y_test,
        'prediction': predictions
    })
    plot_data['error'] = plot_data.prediction - plot_data.truth
    # display(plot_data)

    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle(desc)
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
