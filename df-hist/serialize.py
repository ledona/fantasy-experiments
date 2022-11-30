import os
import tempfile
import logging
from pprint import pformat
from typing import Literal

from xgboost import XGBRegressor
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
# from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
# from skl2onnx import update_registered_converter, to_onnx
# from skl2onnx.common.data_types import FloatTensorType
from sklearn2pmml import sklearn2pmml
import pandas as pd


# update_registered_converter(
#     XGBRegressor, 'XgBoostRegression', 
#     calculate_linear_regressor_output_shapes,
#     convert_xgboost
# )

COL_SEP = '\t'
SUPPORTED_EXPORT_MODELS = ['tpot'] # , 'skautoml']
LOGGER = logging.getLogger("serialize")


def get_tpot_export_code(
    model, 
    data: None | tuple[pd.DataFrame, pd.Series] = None, 
    tmp_path: str | None = None
) -> str:
    """ create the code to export tpot models to ONNX """
    if data is not None:
        X_train, y = data
        df = X_train.copy()
        df['target'] = y
        if tmp_path is None:
            tmp_path = tempfile.gettempdir()
        tpot_data_file = os.path.join(tmp_path, 'tpot-data.csv')
        LOGGER.debug("tpot serialize csv export to '%s'", tpot_data_file)
        # print(df)
        df.to_csv(tpot_data_file, index=False, sep=COL_SEP)
    else:
        tpot_data_file = "no-data-exported.csv"

    # TODO: this is messy af, but exported_pipeline is getting dropped from locals for some reason, so need to assign to another variable in the exec code
    export_code = model.export(data_file_path=tpot_data_file)
    export_code = export_code.replace("COLUMN_SEPARATOR", COL_SEP) + \
        "\nexported_model = exported_pipeline\n"

    # print("###### EXPORT CODE ######")
    # print(export_code)
    # print("#########################")
    # print("running exported code...")

    return export_code


def serialize_tpot(model, X_train, y, name: str | None, tmp_path, desc_folder: str | None, output_format: str):
    assert (desc_folder is None) or (desc_folder is not None and name is not None)
    model_desc_filepath = None
    if output_format == 'onnx':
        export_code = get_tpot_export_code(model, data=(X_train, y), tmp_path=tmp_path)
        # following code should add exported_pipeline to locals
        exec(export_code)

        if 'exported_model' not in locals():
            raise ValueError(
                f"exported_model not defined in locals... keys in local: {locals().keys()}")
        if not (locals()['exported_model']):
            raise ValueError("exported_model is None")
        exported_pipeline = locals()['exported_model']

        if desc_folder:
            model_desc_filepath = os.path.join(desc_folder, name + ".model.py")
            with open(model_desc_filepath, "w") as f:
                f.write(export_code)
        return exported_pipeline, export_code, model_desc_filepath
    elif output_format == 'pmml':
        if desc_folder:
            model_desc_filepath = os.path.join(desc_folder, name + ".model.txt")
            with open(model_desc_filepath, "w") as f:
                f.write(str(model))
        return model, None, model_desc_filepath
    else:
        raise ValueError(f"output_format '{output_format}' not supported")


class SerializeFailure(Exception):
    pass


def get_serialized_file_path(model_name, model_dir, output_format):
    return os.path.join(model_dir, model_name + "." + output_format)

def serialize_model(
    model, model_type,
    X_train: pd.DataFrame, 
    y: pd.Series,
    full_model_name, 
    model_folder=None,
    model_desc_folder=None,
    tmp_path=None,
    output_format: Literal['onnx', 'pmml']='onnx'
):
    LOGGER.info(f"Serializing {full_model_name=} to {output_format=}")

    if model_desc_folder and not os.path.isdir(model_desc_folder):
        os.mkdir(model_desc_folder)
    if model_folder and not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    export_code = None
    try:
        if model_type.startswith('tpot'):
            exported_pipeline, export_code, model_desc_filepath = serialize_tpot(
                model, X_train, y, full_model_name, tmp_path, model_desc_folder, output_format
            )
        elif model_type.startswith('skauto'):
            raise NotImplementedError("skautoml serialization not yet supported. waiting for skautoml to support ONNX 2022.08.25")
            exported_pipeline = model
            if model_desc_folder:
                model_details = pformat(model.show_models())
                model_desc_filepath = os.path.join(model_desc_folder, full_model_name + ".model.txt")
                with open(model_desc_filepath, "w") as f:
                    f.write(model_details)
        else:
            raise NotImplementedError(f"model_type {model_type} not supported. Supported models: {SUPPORTED_EXPORT_MODELS}")

        model_filepath = get_serialized_file_path(full_model_name, model_folder, output_format)

        if output_format == 'onnx':
            serialized_model = to_onnx(exported_pipeline, X=X_train,
                                       name=full_model_name,
                                       final_types=[(y.name, FloatTensorType([1, 1]))])

            if model_folder:
                with open(model_filepath, "wb") as f:
                    f.write(serialized_model.SerializeToString())
                LOGGER.info(f"Exported ONNX model to {model_filepath=}")
        elif output_format == 'pmml':
            serialized_model = exported_pipeline
            if model_folder:
                sklearn2pmml(serialized_model, model_filepath, with_repr=True)
                LOGGER.info(f"Exported PMML model to {model_filepath=}")
        else:
            raise NotImplementedError(f"output_format {output_format} not supported. Supported formats: {['onnx', 'pmml']}")
    except Exception as ex:
        raise SerializeFailure({
            'cause': "Exception during serialization",
            'ex': ex,
            'model': model,
            'output_format': output_format,
            'exported_pipeline': exported_pipeline,
            'name': full_model_name,
            'X': X_train,
            'y': y,
            'final_types': [(y.name, FloatTensorType([1, 1]))],
            'export_code': export_code,
        }) from ex

    return serialized_model, model_filepath