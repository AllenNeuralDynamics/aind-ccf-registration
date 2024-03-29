"""
CCF registration of an image to the Allen Institute's atlas
"""
import logging
import multiprocessing
import os
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Hashable, List, Sequence, Tuple, Union

import ants
import dask
import dask.array as da
import numpy as np
import tifffile
import xarray_multiscale
import zarr
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeZarrWriter
from aind_data_schema.core.processing import DataProcess, ProcessName
from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import Dict as sch_dict
from argschema.fields import Int
from argschema.fields import List as sch_list
from argschema.fields import Str
from dask.distributed import Client, LocalCluster, performance_report
from distributed import wait
from numcodecs import blosc
from skimage import io

from .__init__ import __version__
from .utils import check_orientation, create_folder, generate_processing

blosc.use_threads = False
PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.
    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded
    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data
    Parameters
    ------------------------
    data: dask.array.core.Array
        Dask array of the image data
    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image
    scale_axis: Tuple[int]
        Scaling applied to each axis
    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"
    Returns
    ------------------------
    List[dask.array.core.Array]:
        List with the downsampled image(s)
    """

    pyramid = xarray_multiscale.multiscale(
        data,
        xarray_multiscale.reducers.windowed_mean,  # func
        scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [arr.data for arr in pyramid]


def get_pyramid_metadata() -> dict:
    """
    Gets pyramid metadata in OMEZarr format
    Returns
    ------------------------
    dict:
        Dictionary with the downscaling OMEZarr metadata
    """

    return {
        "metadata": {
            "description": """Downscaling implementation based on the
                windowed mean of the original array""",
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": str(xarray_multiscale.__version__),
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


class RegSchema(ArgSchema):
    """
    Schema format for Registration.
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

    reference = Str(
        metadata={"required": True, "description": "Reference image"}
    )

    output_data = Str(
        metadata={"required": True, "description": "Output file"}
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

    ants_params = sch_dict(
        metadata={
            "required": True,
            "description": "ants registering parameters",
        }
    )

    downsampled_file = Str(
        metadata={"required": True, "description": "Downsampled file"}
    )

    downsampled16bit_file = Str(
        metadata={"required": True, "description": "Downsampled 16bit file"}
    )

    reference_res = Int(
        metadata={
            "required": True,
            "description": "Voxel Resolution of reference in microns",
        }
    )

    affine_transforms_file = Str(
        metadata={
            "required": True,
            "description": "Output forward affine Transforms file",
        }
    )

    ls_ccf_warp_transforms_file = Str(
        metadata={
            "required": True,
            "description": "Output inverse warp Transforms file",
        }
    )

    ccf_ls_warp_transforms_file = Str(
        metadata={
            "required": True,
            "description": "Output forward warp Transforms file",
        }
    )


class Register(ArgSchemaParser):
    """
    Class to Register lightsheet data to CCF atlas
    """

    default_schema = RegSchema

    def __read_zarr_image(self, image_path: PathLike) -> np.array:
        """
        Reads a zarr image

        Parameters
        -------------
        image_path: PathLike
            Path where the zarr image is located

        Returns
        -------------
        np.array
            Numpy array with the zarr image
        """
        image_path = str(image_path)
        zarr_img = zarr.open(image_path, mode="r")
        img_array = np.asarray(zarr_img)
        img_array = np.squeeze(img_array)
        return img_array

    def atlas_alignment(
        self, img_array: np.array, ants_params: dict
    ) -> np.array:
        """
        Aligns the image to the reference atlas

        Parameters
        ------------

        img_array: np.array
            Array with the image

        ants_params: dict
            Dictionary with ants parameters
        """
        # get data orientation
        img_array = img_array.astype(np.double)
        img_out, in_mat, out_mat = check_orientation(
            img_array,
            self.args["input_orientation"],
            self.args["ants_params"]["orientations"],
        )

        logger.info(
            f"Input image dimensions: {img_array.shape} \nInput image orientation: {in_mat}"
        )
        logger.info(
            f"Output image dimensions: {img_out.shape} \nOutput image orientation: {out_mat}"
        )

        # convert input data to tiff into reference voxel resolution
        ants_img = ants.from_numpy(img_out, spacing=ants_params["spacing"])
        fillin = ants.resample_image(
            ants_img, ants_params["new_spacing"], False, 1
        )
        logger.info(f"Size of resampled image: {fillin.shape}")

        downsampled_file_path = Path(
            f"{self.args['metadata_folder']}/{self.args['downsampled_file']}"
        )
        ants.image_write(fillin, str(downsampled_file_path))

        # convert data to uint16
        im = io.imread(str(downsampled_file_path)).astype(np.uint16)

        downsampled16bit_file_path = Path(
            f"{self.args['metadata_folder']}/{self.args['downsampled16bit_file']}"
        )

        tifffile.imwrite(str(downsampled16bit_file_path), im)
        # read images
        logger.info("Reading reference image")
        img1 = ants.image_read(os.path.abspath(self.args["reference"]))
        img2 = ants.image_read(str(downsampled16bit_file_path))

        # register with ants
        reg12 = ants.registration(
            img1, img2, "SyN", reg_iterations=[100, 10, 0]
        )

        # output
        shutil.copy(
            reg12["fwdtransforms"][0],
            self.args["ccf_ls_warp_transforms_file"],
        )
        shutil.copy(
            reg12["fwdtransforms"][1],
            self.args["affine_transforms_file"],
        )
        shutil.copy(
            reg12["invtransforms"][1],
            self.args["ls_ccf_warp_transforms_file"],
        )

        return reg12["warpedmovout"].numpy()

    def write_zarr(
        self,
        img_array: np.array,
        physical_pixel_sizes: List[int],
        output_path: PathLike,
        image_name: PathLike,
        opts: dict,
    ):
        """
        Writes array to the OMEZarr format

        Parameters
        ------------
        img_array: dask.Array.core
            Array with the registered image

        physical_pixel_sizes: List[int]
            List with the physical pixel sizes.
            The order must be [Z, Y, X]

        output_path: PathLike
            Path where the .zarr image will be written

        image_name: PathLike
            Image name for the .zarr image

        opts: dict
            Dictionary with the storage
            options for the zarr image
        """

        dask_folder = Path("../scratch")
        # Setting dask configuration
        dask.config.set(
            {
                "temporary-directory": dask_folder,
                "local_directory": dask_folder,
                "tcp-timeout": "300s",
                "array.chunk-size": "384MiB",
                "distributed.comm.timeouts": {
                    "connect": "300s",
                    "tcp": "300s",
                },
                "distributed.scheduler.bandwidth": 100000000,
                "distributed.worker.memory.rebalance.measure": "optimistic",
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": 0.92,
                "distributed.worker.memory.pause": 0.95,
                "distributed.worker.memory.terminate": 0.98,
            }
        )

        physical_pixels = PhysicalPixelSizes(
            physical_pixel_sizes[0],
            physical_pixel_sizes[1],
            physical_pixel_sizes[2],
        )

        scale_axis = [2, 2, 2]
        pyramid_data = compute_pyramid(
            img_array,
            -1,
            scale_axis,
            self.args["OMEZarr_params"]["chunks"],
        )

        pyramid_data = [pad_array_n_d(pyramid) for pyramid in pyramid_data]
        logger.info(f"Pyramid {pyramid_data}")

        # Writing OMEZarr image

        n_workers = multiprocessing.cpu_count()
        threads_per_worker = 1
        # Using 1 thread since is in single machine.
        # Avoiding the use of multithreaded due to GIL

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit="auto",
        )
        client = Client(cluster)

        writer = OmeZarrWriter(output_path)

        dask_report_file = Path(self.args["metadata_folder"]).joinpath(
            "dask_report.html"
        )

        with performance_report(filename=dask_report_file):
            dask_jobs = writer.write_multiscale(
                pyramid=pyramid_data,
                image_name=image_name,
                chunks=pyramid_data[0].chunksize,
                physical_pixel_sizes=physical_pixels,
                channel_names=None,
                channel_colors=None,
                scale_factor=scale_axis,
                storage_options=opts,
                compute_dask=False,
                **get_pyramid_metadata(),
            )

            if len(dask_jobs):
                dask_jobs = dask.persist(*dask_jobs)
                wait(dask_jobs)

        client.close()

    def run(self) -> str:
        """
        Runs CCF registration
        """

        # Creating output folders
        input_data_path = os.path.abspath(self.args["input_data"])
        output_data_path = os.path.abspath(self.args["output_data"])
        metadata_path = os.path.abspath(self.args["metadata_folder"])
        reference_path = os.path.abspath(self.args["reference"])
        # input_data_path = glob(f"{input_data_path}_stitched_*/")[0]

        logger.info(
            f"Input data: {input_data_path}\nOutput data: {output_data_path}\nMetadata path: {metadata_path} reference: {reference_path}"
        )

        create_folder(output_data_path)
        create_folder(metadata_path)

        # read input data (lazy loading)
        # flake8: noqa: E501
        image_path = Path(input_data_path).joinpath(
            f"{self.args['input_channel']}.zarr/{self.args['input_scale']}"
        )
        logger.info(f"Going to read zarr: {image_path}")

        data_processes = []

        if not os.path.isdir(str(image_path)):
            root_path = Path(input_data_path)
            channels = [
                folder
                for folder in os.listdir(root_path)
                if folder != ".zgroup"
            ]

            selected_channel = channels[0]

            logger.info(
                f"""Directory {image_path} does not exist!
                Setting registration to the first available channel: {selected_channel}"""
            )
            image_path = root_path.joinpath(
                f"{selected_channel}/{self.args['input_scale']}"
            )

        start_date_time = datetime.now()
        img_array = self.__read_zarr_image(image_path)
        end_date_time = datetime.now()

        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_IMPORTING,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(image_path),
                outputs={},
                code_url=self.args["code_url"],
                code_version=__version__,
                parameters={},
                notes="Importing fused data for alignment",
            )
        )

        # Atlas alignment
        start_date_time = datetime.now()
        ants_params = self.args["ants_params"]
        ants_params["new_spacing"] = (
            self.args["reference_res"],
            self.args["reference_res"],
            self.args["reference_res"],
        )
        ants_params["reference"] = reference_path
        aligned_image = self.atlas_alignment(img_array, ants_params)
        end_date_time = datetime.now()

        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_ATLAS_ALIGNMENT,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(image_path),
                outputs={},
                code_url="https://github.com/ANTsX/ANTs",
                code_version=ants.__version__,
                parameters=ants_params,
                notes="Registering image data to Allen CCF Atlas",
            )
        )

        start_date_time = datetime.now()
        image_name = "image.zarr"

        opts = {
            "compressor": blosc.Blosc(
                cname=self.args["OMEZarr_params"]["compressor"],
                clevel=self.args["OMEZarr_params"]["clevel"],
                shuffle=blosc.SHUFFLE,
            )
        }

        aligned_image_dask = da.from_array(aligned_image)
        self.write_zarr(
            img_array=aligned_image_dask,  # dask array
            physical_pixel_sizes=ants_params["new_spacing"],
            output_path=output_data_path,
            image_name=image_name,
            opts=opts,
        )
        end_date_time = datetime.now()

        data_processes.append(
            DataProcess(
                name=ProcessName.FILE_CONVERSION,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location="In memory array",
                output_location=str(
                    Path(output_data_path).joinpath(image_name)
                ),
                outputs={},
                code_url=self.args["code_url"],
                code_version=__version__,
                parameters={
                    "pixel_sizes": ants_params["new_spacing"],
                    "OMEZarr_params": self.args["OMEZarr_params"],
                },
                notes="Converting registered image to OMEZarr",
            )
        )

        processing_path = Path(metadata_path).joinpath("processing.json")

        logger.info(f"Writing processing: {processing_path}")

        generate_processing(
            data_processes=data_processes,
            dest_processing=metadata_path,
            processor_full_name="Camilo Laiton",
            pipeline_version="1.5.0",
        )

        return str(image_path)


def main(input_config: dict):
    """
    Main function to execute
    """

    mod = Register(input_config)
    return mod.run()


if __name__ == "__main__":
    main()
