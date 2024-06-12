"""
This config file points to data directories, defines global variables,
specify schema format for Preprocess and Registration.
"""

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
from argschema import ArgSchema
from argschema.fields import Dict as sch_dict
from argschema.fields import Int
from argschema.fields import List as sch_list
from argschema.fields import Str

PathLike = Union[str, Path]
ArrayLike = Union[da.core.Array, np.ndarray]

VMIN = 0
VMAX = 1.5


class RegSchema(ArgSchema):
    """
    Schema format for Preprocess and Registration.
    """

    input_data = Str(
        metadata={
            "required": True,
            "description": "Input data without timestamp",
        }
    )

    input_channel = Str(
        metadata={"required": True, "description": "Channel to register"}
    )

    input_scale = Int(
        metadata={"required": True, "description": "Zarr scale to start with"}
    )

    input_orientation = sch_list(
        cls_or_instance=sch_dict,
        metadata={
            "required": True,
            "description": "Brain orientation during aquisition",
        },
    )

    template_path = Str(
        metadata={"required": True, "description": "Path to the SPIM template"}
    )

    ccf_reference_path = Str(
        metadata={"required": True, "description": "Path to the CCF template"}
    )

    template_to_ccf_transform_path = sch_list(
        cls_or_instance=Str,
        metadata={
            "required": True,
            "description": "Path to the template-to-ccf transform",
        },
    )

    ccf_annotation_to_template_moved_path = Str(
        metadata={
            "required": True,
            "description": "Path to CCF annotation in SPIM template space",
        }
    )

    output_data = Str(
        metadata={"required": True, "description": "Output file"}
    )

    results_folder = Str(
        metadata={
            "required": True,
            "description": "Folder to save registration results",
        }
    )

    reg_folder = Str(
        metadata={
            "required": True,
            "description": "Folder to save derivative results of registration",
        }
    )

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )

    code_url = Str(
        metadata={"required": True, "description": "CCF registration URL"}
    )

    metadata_folder = Str(
        metadata={"required": True, "description": "Metadata folder"}
    )

    OMEZarr_params = sch_dict(
        metadata={
            "required": True,
            "description": "OMEZarr writing parameters",
        }
    )

    prep_params = sch_dict(
        metadata={
            "required": True,
            "description": "raw data preprocessing parameters",
        }
    )

    ants_params = sch_dict(
        metadata={
            "required": True,
            "description": "ants registering parameters",
        }
    )

    reference_res = Int(
        metadata={
            "required": True,
            "description": "Voxel Resolution of reference in microns",
        }
    )
