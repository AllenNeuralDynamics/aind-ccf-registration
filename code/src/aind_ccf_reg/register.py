"""
CCF registration of an image to the Allen Institute's atlas
"""
import logging
import os
from pathlib import Path
from typing import Union

import ants
import numpy as np
import tifffile
import zarr
from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import Int, Str
from skimage import io
import shutil

PathLike = Union[str, Path]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RegSchema(ArgSchema):
    """
    Schema format for Registration.
    """

    input_data = Str(metadata={"required": True, "description": "Input data"})

    input_channel = Str(
        metadata={"required": True, "description": "Channel to register"}
    )

    input_zarr_directory = Str(
        metadata={
            "required": True,
            "description": "directory with Ome zarr data",
        }
    )

    input_scale = Int(
        metadata={"required": True, "description": "Zarr scale to start with"}
    )

    reference = Str(
        metadata={"required": True, "description": "Reference image"}
    )

    output_data = Str(
        metadata={"required": True, "description": "Output file"}
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

    fwd_transforms_file = Str(
        metadata={
            "required":True, 
            "description":"Output forward Transforms file",
        }
    )
    
    inv_transforms_file = Str(
        metadata={
            "required":True, 
            "description":"Output inverse Transforms file",
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

    def run(self) -> str:
        """
        Runs CCF registration
        """
        # read input data (lazy loading)
        # flake8: noqa: E501
        image_path = Path(self.args["input_data"]).joinpath(
            f"{self.args['input_zarr_directory']}/{self.args['input_channel']}/{self.args['input_scale']}"
        )
        logger.info(f"Going to read zarr: {image_path}")

        if not os.path.isdir(str(image_path)):

            root_path = Path(self.args["input_data"]).joinpath(
                self.args["input_zarr_directory"]
            )
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

        img_array = self.__read_zarr_image(image_path)

        # get data orientation
        img_array = img_array.astype(np.double)
        img_array = np.swapaxes(img_array, 0, 2)
        img_array = np.swapaxes(img_array, 1, 2)
        img_array = np.flip(img_array, 2)

        # convert input data to tiff into reference voxel resolution
        ants_img = ants.from_numpy(img_array, spacing=(14.4, 14.4, 16))
        new_spacing = (25, 25, 25)
        fillin = ants.resample_image(ants_img, new_spacing, False, 1)
        logger.info(f"Size of resampled image: {fillin.shape}")

        ants.image_write(fillin, self.args["downsampled_file"])

        # convert data to uint16
        im = io.imread(self.args["downsampled_file"]).astype(np.uint16)
        tifffile.imwrite(self.args["downsampled16bit_file"], im)

        # read images
        logger.info("Reading reference image")
        img1 = ants.image_read(self.args["reference"])
        img2 = ants.image_read(self.args["downsampled16bit_file"])

        # register with ants
        reg12 = ants.registration(
            img1, img2, "SyN", reg_iterations=[100, 10, 0]
        )

        # output
        ants.image_write(
            reg12["warpedmovout"],
            "%s_%dmicrons.tiff"
            % (self.args["output_data"], self.args["reference_res"]),
        )
        shutil.copy(
            reg12['fwdtransforms'][0], 
            self.args['fwd_transforms_file'],
        )
        shutil.copy(
            reg12['invtransforms'][0], 
            self.args['inv_transforms_file'],
        )

        return str(image_path)


def main():
    """
    Main function to execute
    """
    example_input = {
        "reference": "/data/reference.tiff",
        "reference_res": 25,
        "output_data": "/results/registered_to_atlas",
        "downsampled_file": "/results/downsampled.tiff",
        "downsampled16bit_file": "/results/downsampled_16.tiff",
        "fwd_transforms_file": "/results/fwd_transforms.nii.gz",
        "inv_transforms_file": "/results/inv_transforms.nii.gz",
    }

    mod = Register(example_input)
    return mod.run()


if __name__ == "__main__":
    main()
