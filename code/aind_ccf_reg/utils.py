"""
File for utilities
"""
import os
from pathlib import Path
from typing import List, Optional, Union

import pydantic
from aind_data_schema import Processing

PathLike = Union[str, Path]


def read_json_from_pydantic(
    path: PathLike, pydantic_class: pydantic.main.ModelMetaclass
) -> pydantic.BaseModel:
    """
    Reads a json file and parses it to
    a pydantic model

    path: PathLike
        Path where the json is stored

    pydantic_class: pydantic.main.ModelMetaClass
        Defined pydantic model for the json

    Returns
    -------------
    pydantic.BaseModel
        Model with populated data
    """

    json_data = pydantic.parse_file_as(path=path, type_=pydantic_class)

    return json_data


def generate_processing(
    data_processes: List[dict],
    dest_processing: PathLike,
    pipeline_version: str,
) -> None:
    """
    Generates data description for the output folder.
    Parameters
    ------------------------
    data_processes: List[dict]
        List with the processes aplied in the pipeline.
    dest_processing: PathLike
        Path where the processing file will be placed.
    pipeline_version: str
        Terastitcher pipeline version
    """

    # flake8: noqa: E501
    processing = Processing(
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-ccf-registration",
        pipeline_version=pipeline_version,
        data_processes=data_processes,
    )

    with open(dest_processing, "w") as f:
        f.write(processing.json(indent=3))


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.
    Raises
    ------------------------
    OSError:
        if the folder exists.
    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
