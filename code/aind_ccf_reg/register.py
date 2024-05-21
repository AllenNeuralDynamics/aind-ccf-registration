"""
Register an lightsheet data to the Allen Institute's CCF atlas via the SPIM template

Pipeline:
(1) preprocessing an brain image
(2) rigid + SyN registration: register the preprocessed brain image to the SPIM template 
(3) apply transform to align deformed image from (2) to the CCF template

Quality control on registration:
(1) visualization: plot the overlay/difference image between deformed and fixed images
(2) TODO: compute the similarity metics to automatically detect registration failure

# TODO: need to pass transforms that aligns brain image to CCF to aind-smartspim-cell-quantification capsule
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
from argschema import ArgSchemaParser
from dask.distributed import Client, LocalCluster, performance_report
from distributed import wait
from numcodecs import blosc
from skimage import io


from .__init__ import __version__
blosc.use_threads = False

from aind_ccf_reg.utils import check_orientation, create_folder, generate_processing
from aind_ccf_reg.configs import PathLike, ArrayLike
from aind_ccf_reg.configs import VMIN, VMAX
from aind_ccf_reg.configs import RegSchema
from aind_ccf_reg.plots import plot_antsimgs, plot_reg, plot_reg_before_after
from aind_ccf_reg.preprocess import perc_normalization
from aind_ccf_reg.preprocess import write_and_plot_image
from aind_ccf_reg.preprocess import Preprocess

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

    def _qc_reg(self, ants_moving, ants_fixed, ants_moved, moved_path: PathLike=None, figpath_name: str="reg") -> None:
        """ 
        Quality control on registration results and write deformed image. 
        The plots will be saved to the same folder as the deformed image.
        
        Parameters
        -------------
        ants_fixed: ANTsImage
            fixed image
        ants_moving: ANTsImage
            moving image
        ants_moved: ANTsImage
            deformed image
        moved_path: PathLike
            Path to save deformed image
        figpath_name: str
            figpath name
        """
        # plot moving, fixed, moved, overlaid, difference images in three directions
        figpath = f"{self.args['reg_folder']}/{figpath_name}" 
        logger.info(f"Plot registration results: {figpath}")   
        
        if np.any(ants_moving.direction != ants_fixed.direction):
            logger.info(f"Reorient moving image direction to fixed image direction ...")    
            ants_moving = ants.reorient_image2(ants_moving, orientation=ants.get_orientation(ants_fixed))   
            logger.info(f"Reoriented moving image -- {ants_moving}")    

        for loc in [0, 1, 2]:        
            plot_args = (ants_moving, ants_fixed, ants_moved, f"{figpath}_{loc}")
            plot_kwargs = {"title": figpath_name, "loc": loc, "vmin": VMIN, "vmax": VMAX} 

            if np.all(ants_moving.shape == ants_fixed.shape):
                # moving and fixed images have same dimension
                plot_reg_before_after(*plot_args, **plot_kwargs)
            else:
                # moving and fixed images do not have same dimension
                plot_reg(*plot_args, **plot_kwargs)
            
        # plot moved image
        if moved_path:
            figpath = moved_path.replace(".nii.gz", "")
            title = moved_path.replace(".nii.gz", "").split("/")[-1]
            logger.info(f"Plot aligned image: {figpath}, title: {title}")    
            plot_antsimgs(ants_moved, figpath, title, vmin=VMIN, vmax=VMAX)
        
            logger.info(f"Saving aligned image: {moved_path}")    
            ants.image_write(ants_moved, moved_path) 
            logger.info("Done saving")

    def rigid_register(self, ants_fixed, ants_moving):
        """ Run rigid regsitration to align brain image to SPIM template
        
        Parameters
        -------------
        ants_fixed: ANTsImage
            fixed image
        ants_moving: ANTsImage
            moving image
        
        Returns
        -----------
        ANTsImage
            deformed image
        """
        logger.info(f"\nStart computing rigid registration ....")

        # run registration
        start_time = datetime.now()
        registration_params = {
            "fixed": ants_fixed,
            "moving": ants_moving, 
            "type_of_transform": "Rigid",
            "outprefix": f"{self.args['reg_folder']}/rigid_", 
            "mask_all_stages": True,
            "grad_step": 0.25,
            "reg_iterations": (60, 40, 20, 0),
            "aff_metric": "mattes"            
        }

        logger.info(f"Computing rigid registration with parameters: {registration_params}")
        reg = ants.registration(**registration_params)
        end_time = datetime.now()
        logger.info(f"Rigid registration Complete, execution time: {end_time - start_time} s -- image {reg}")

        ants_moved = reg["warpedmovout"]
        
        reg_task = "reg_rigid"
        self._qc_reg(ants_moving, 
                     ants_fixed, 
                     ants_moved, 
                     moved_path=self.args["ants_params"]["rigid_path"], 
                     figpath_name=reg_task)

        return ants_moved

    
    def register_to_template(self, ants_fixed, ants_moving):  
        """ 
        Run SyN regsitration to align brain image to SPIM template
        
        Parameters
        -------------
        ants_fixed: ANTsImage
            fixed image
        ants_moving: ANTsImage
            moving image
        
        Returns
        -----------
        ANTsImage
            deformed image
        """
        logger.info(f"\nStart registering to template ....")

        if self.args['reference_res'] == 25:
            reg_iterations = [100, 10, 0] # TODO
            # reg_iterations = [1, 0, 0, 0]
        elif self.args['reference_res'] == 10:
            reg_iterations = [400, 200, 40, 0]
        else:
            raise ValueError(
                f"Resolution {self.args['reference_res']} is not allowed. Allowed values are: 10, 25"
            )
            
        start_time = datetime.now()
        registration_params = {
            "fixed": ants_fixed,
            "moving": ants_moving,
            "syn_metric": "CC",                     
            "syn_sampling": 2,
            "reg_iterations": reg_iterations, 
            "outprefix": f"{self.args['reg_folder']}/"}
        
        logger.info(f"Computing SyN registration with parameters: {registration_params}")
        reg = ants.registration(**registration_params)
        end_time = datetime.now()
        logger.info(f"SyN registration complete, execution time: {end_time - start_time} s -- image {reg}")
        
        ants_moved = reg["warpedmovout"]
        
        reg_task = "reg_to_template"
        self._qc_reg(ants_moving, 
                     ants_fixed, 
                     ants_moved, 
                     moved_path=self.args["ants_params"]["moved_to_template_path"], 
                     figpath_name=reg_task)
        
        return ants_moved
    
    def register_to_ccf(self, ants_fixed, ants_moving):  
        """ 
        Run manual regsitration to align brain image to CCF template
        
        Parameters
        -------------
        ants_fixed: ANTsImage
            fixed image
        ants_moving: ANTsImage
            moving image
        
        Returns
        -----------
        ANTsImage
            deformed image
        """
        logger.info("\nStart registering to CCF ....")
        logger.info(f"Register to CCF with: {self.args['template_to_ccf_transform_path']}")

        # for visualizing registration results
        ants_fixed = perc_normalization(ants_fixed)
        
        start_time = datetime.now()
        ants_moved = ants.apply_transforms(
                fixed  = ants_fixed,
                moving = ants_moving ,
                transformlist=self.args["template_to_ccf_transform_path"] 
            )
        end_time = datetime.now()

        logger.info(f"Register to CCF, execution time: {end_time - start_time} s -- image {ants_moved}")

        reg_task = "reg_to_ccf"
        self._qc_reg(ants_moving, 
                     ants_fixed, 
                     ants_moved, 
                     moved_path=self.args["ants_params"]["moved_to_ccf_path"], 
                     figpath_name=reg_task)
        
        return ants_moved
    
    
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
        #----------------------------------#
        # load SPIM template + CCF
        #----------------------------------#
    
        logger.info("Reading reference images")
        ants_template = ants.image_read(os.path.abspath(self.args["template_path"])) # SPIM template
        ants_ccf      = ants.image_read(os.path.abspath(self.args["ccf_reference_path"])) # CCF template 
        logger.info(f"Loaded SPIM template {ants_template}")
        logger.info(f"Loaded CCF template {ants_ccf}")
        
        
        #----------------------------------#
        # orient data to SPIM template's direction
        #----------------------------------#

        img_array = img_array.astype(np.double)
        img_out, in_mat, out_mat = check_orientation(
            img_array,
            self.args["input_orientation"],
            self.args["ants_params"]["template_orientations"],
        )

        logger.info(
            f"Input image dimensions: {img_array.shape} \nInput image orientation: {in_mat}"
        )
        logger.info(
            f"Output image dimensions: {img_out.shape} \nOutput image orientation: {out_mat}"
        )

        ants_img = ants.from_numpy(img_out, spacing=ants_params["spacing"])
        ants_img.set_direction(ants_template.direction)
        ants_img.set_origin(ants_template.origin)
        
        write_and_plot_image(
            ants_img,
            data_path=self.args["prep_params"].get("rawdata_path"), 
            plot_path=self.args["prep_params"].get("rawdata_figpath"), vmin=0, vmax=500)
    
        #----------------------------------#
        # run preprocessing on raw data
        #----------------------------------#

        prep = Preprocess(self.args, ants_img, ants_template)
        ants_img = prep.run()
        logger.info(f"Preprocessed input data {ants_img}")
        
        #----------------------------------#
        # register brain image to CCF
        #----------------------------------#
        # ants_img = ants.image_read(self.args["prep_params"].get("percNorm_path")) # 

        # register to SPIM template: rigid + SyN
        rigid_image   = self.rigid_register(ants_template, ants_img)
        aligned_image = self.register_to_template(ants_template, rigid_image)
        
        # aligned_image = ants.image_read(self.args["ants_params"].get("moved_to_template_path")) # 
        
        # register to CCF template: apply manual regsitration
        aligned_image = self.register_to_ccf(ants_ccf, aligned_image)

        return aligned_image.numpy()

    
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
        reg_folder = os.path.abspath(self.args["reg_folder"]) # save registration results
        # input_data_path = glob(f"{input_data_path}_stitched_*/")[0]

        logger.info(
            f"Input data: {input_data_path}\nOutput data: {output_data_path}\nMetadata path: {metadata_path}"
        )
        
        logger.info(f"Regsitration results save to: {reg_folder}")
        
        create_folder(output_data_path)
        create_folder(reg_folder)
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

        aligned_image = self.atlas_alignment(img_array, ants_params)
        end_date_time = datetime.now()

        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_ATLAS_ALIGNMENT,
                software_version=__version__,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                input_location=str(image_path),
                output_location=str(reg_folder),
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
            processor_full_name="Di Wang, Camilo Laiton",
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
