"""
File for utilities
"""
import os
import numpy as np

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

def check_orientation(img: np.array, params: dict, orientations: dict):
    """
    Checks aquisition orientation an makes sure it is aligned to the CCF. The
    CCF orientation is:
        - superior_to_inferior
        - left_to_right
        - anterior_to_posterior

    Parameters
    ----------
    img : np.array
        The raw image in its aquired orientatin
    params : dict
        The orientation information from processing_manifest.json
    orientations: dict
        The axis order of the CCF reference atals

    Returns
    -------
    img_out : np.array
        The raw image oriented to the CCF
    """
    
    orient_mat= np.zeros((3,3))
    acronym = ['', '', '']
    
    
    for k, vals in enumerate(params):
        direction = vals['direction'].lower()
        dim = vals['dimension']
        if direction in orientations.keys():
            ref_axis = orientations[direction]
            orient_mat[dim, ref_axis] = 1
            acronym[dim] = direction[0]
        else: 
            direction_flip = '_'.join(direction.split('_')[::-1])
            ref_axis = orientations[direction_flip]
            orient_mat[dim, ref_axis] = -1
            acronym[dim] = direction[0]
    
    #check because there was a bug that allowed for invalid spl orientation
    #all vals should be postitive so just taking absolute value of matrix
    if "".join(acronym) == 'spl':
        orient_mat = abs(orient_mat)
        
    
    original, swapped = np.where(orient_mat)
    img_out = np.moveaxis(img, original, swapped)
    
    out_mat = orient_mat[:, swapped]
    for c, row in enumerate(orient_mat.T):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
            out_mat[val, val] *= -1
            
    return img_out, orient_mat, out_mat
