"""
Preprocess lightsheet data
"""

import logging
from datetime import datetime

import ants
import numpy as np
import scipy.ndimage as ni
from aind_ccf_reg.configs import VMAX, VMIN
from aind_ccf_reg.plots import plot_antsimgs
from skimage.filters import threshold_li
from skimage.measure import label

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def invert_perc_normalization(ants_img, percentile_values):
    """
    Inverse of the percentile normalization

    Parameters
    ----------
    ants_img: ANTsImage
        Normalized image

    Returns
    -------
    ANTsImage
    """
    p0, p1 = percentile_values[0], percentile_values[1]
    new_img = (ants_img * (p1 - p0)) + p0
    return new_img


def perc_normalization(
    ants_img, lower_perc: float = 2, upper_perc: float = 98
):
    """
    Percentile Normalization

    Parameters
    -------------
    ants_img: ANTsImage

    Returns
    -----------
    ANTsImage
    """
    percentiles = [lower_perc, upper_perc]
    percentile_values = np.percentile(ants_img.view(), percentiles)
    assert percentile_values[1] > percentile_values[0]

    ants_img = np.maximum(ants_img, percentile_values[0])

    ants_img = (ants_img - percentile_values[0]) / (
        percentile_values[1] - percentile_values[0]
    )

    return ants_img, percentile_values


def write_and_plot_image(
    ants_img, data_path=None, plot_path=None, vmin=VMIN, vmax=VMAX
):
    """
    Write and plot an ANTsImage

    Parameters
    -------------
    ants_img: ANTsImage
        image will be plotted or/and saved.
    data_path: PathLike
        Path where the ANTsImage to be saved
    plot_path: PathLike
        Path where the plot of ANTsImage to be saved
    vmin, vmax: float
        Set the color limits of the antsImage.
    """
    if plot_path:
        title = plot_path.split("/")[-1].split(".")[0]
        plot_antsimgs(ants_img, plot_path, title, vmin=vmin, vmax=vmax)

    if data_path:
        ants.image_write(ants_img, data_path)


class Masking:
    """
    Get a binary mask image from the given light sheet volume after
    thresholding. We compute the optimal threshold using Li thresholding
    """

    def __init__(self, ants_img):
        """Class constructor
        Parameters
        ----------
        ants_img: ANTsImage
            image from which mask will be computed.
        """
        self.ants_img = ants_img

    def _getLargestCC(self, segmentation):
        """get the largest connected component"""
        labels = label(segmentation)
        assert labels.max() != 0  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

        return largestCC

    def _get_threshold_li(self, arr_img: np.ndarray) -> float:
        """get the optimal threshold using Li thresholding"""
        start_time = datetime.now()
        low_thresh = threshold_li(arr_img)
        end_time = datetime.now()

        logger.info(
            f"Find optimal threshold using Li thresholding, execution time:\
            {end_time - start_time} s -- low_thresh={low_thresh}"
        )
        return low_thresh

    def _cleanup_mask(self, arr_mask: np.ndarray) -> np.ndarray:
        """
        Morphological operations will be applied to clean up the mask by
        closing holes and eroding away small or weakly-connected areas.
        The following steps are applied:
            - Closing holes
            - Dilation with radius 1 voxel
            - Morphological closing
            - Retain largest component
        """
        # 3x3 structuring element with connectivity 2
        struct = ni.generate_binary_structure(3, 2)

        mask = ni.binary_fill_holes(arr_mask).astype(int)
        mask = ni.binary_dilation(mask, structure=struct).astype(int)
        mask = ni.binary_closing(mask).astype(int)
        mask = self._getLargestCC(mask)

        mask = ni.binary_dilation(mask, structure=struct, iterations=6).astype(
            int
        )
        arr_mask = ni.binary_fill_holes(mask).astype(int)

        return arr_mask

    def run(self) -> np.ndarray:
        """compute the mask"""
        arr_img = self.ants_img.numpy()

        # get optimal threshold using Li thresholding
        # https://scikit-image.org/docs/stable/auto_examples/developers/plot_threshold_li.html
        low_thresh = self._get_threshold_li(arr_img)

        # thresholding
        arr_mask = arr_img > low_thresh

        # clean up
        arr_mask = self._cleanup_mask(arr_mask)

        # convert numpy array to ants image
        ants_img_mask = ants.from_numpy(
            arr_mask.astype("float32"),
            spacing=self.ants_img.spacing,
            origin=self.ants_img.origin,
            direction=self.ants_img.direction,
        )

        return ants_img_mask


class Preprocess:
    """
    Class to Preprocess lightsheet data
    1. resample to isotropic to have same resolution of the SPIM template
    2. N4 bias correction
    3. intensity normalization
    """

    def __init__(self, args, input_data, reference_data):
        """Initialize Preprocess class.
        Parameters
        ----------
        args: dict
            Dictionary with preprocessing parameters
        input_data: ANTsImage
            image to be processed
        reference_data: ANTsImage
            spim template
        """
        self.args = args
        self.input_data = input_data
        self.reference_data = reference_data

    def resample(self, ants_img, ants_template):
        """Resample OMEZarr image to the resolution of template"""
        logger.info("Resample OMEZarr image to the resolution of template")
        ants_img = ants.resample_image(
            ants_img, ants_template.spacing, False, 1
        )

        logger.info(f"Resampled OMEZarr dataset: {ants_img}")

        write_and_plot_image(
            ants_img,
            data_path=self.args["prep_params"].get("resample_path"),
            plot_path=self.args["prep_params"].get("resample_figpath"),
            vmin=0,
            vmax=500,
        )

        return ants_img

    def compute_mask(self, ants_img):
        """compute make"""
        logger.info("Computing Mask")

        start_time = datetime.now()
        mask = Masking(ants_img)
        ants_img_mask = mask.run()
        end_time = datetime.now()

        logger.info(
            f"Mask Complete, execution time: {end_time - start_time} s\
            -- image {ants_img_mask}"
        )

        write_and_plot_image(
            ants_img_mask,
            data_path=self.args["prep_params"].get("mask_path"),
            plot_path=self.args["prep_params"].get("mask_figpath"),
            vmin=0,
            vmax=1,
        )

        return ants_img_mask

    def compute_N4(
        self,
        ants_img,
        mask=None,
        rescale_intensities=False,
        shrink_factor=4,
        convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
        spline_param=None,
        return_bias_field=False,
        verbose=False,
        weight_mask=None,
    ):
        """N4 Bias Field Correction
        https://antspy.readthedocs.io/en/latest/utils.html#ants.utils.bias_correction.n4_bias_field_correction
        """
        logger.info("Computing N4")
        n4_bias_params = {
            "mask": mask,
            "rescale_intensities": rescale_intensities,
            "shrink_factor": shrink_factor,
            "convergence": convergence,
            "spline_param": spline_param,
            "return_bias_field": return_bias_field,
            "verbose": verbose,
            "weight_mask": weight_mask,
        }

        logger.info(f"Parameters -> {n4_bias_params}")
        start_time = datetime.now()
        ants_img_n4 = ants.utils.n4_bias_field_correction(
            ants_img, **n4_bias_params
        )
        end_time = datetime.now()

        logger.info(
            f"N4 Complete, execution time: {end_time - start_time} s\
            -- image {ants_img_n4}"
        )

        write_and_plot_image(
            ants_img_n4,
            data_path=self.args["prep_params"].get("n4bias_path"),
            plot_path=self.args["prep_params"].get("n4bias_figpath"),
            vmin=0,
            vmax=500,
        )

        # Compute the difference between ants_img and ants_img_n4
        ants_img_intensity_difference = ants_img - ants_img_n4

        write_and_plot_image(
            ants_img_intensity_difference,
            data_path=self.args["prep_params"].get("img_diff_n4bias_path"),
            plot_path=self.args["prep_params"].get("img_diff_n4bias_figpath"),
            vmin=0,
            vmax=200,
        )

        return ants_img_n4

    def intensity_norm(self, ants_img):
        """compute percential normalization"""
        logger.info("Start intensity normalization")
        start_time = datetime.now()
        ants_img, percentile_values = perc_normalization(ants_img)
        end_time = datetime.now()
        logger.info(
            f"Intensity normalization complete, execution time:\
            {end_time - start_time} s -- image {ants_img}"
        )

        write_and_plot_image(
            ants_img,
            data_path=self.args["prep_params"].get("percNorm_path"),
            plot_path=self.args["prep_params"].get("percNorm_figpath"),
            vmin=VMIN,
            vmax=VMAX,
        )

        return ants_img, percentile_values

    def run(self) -> str:
        """
        Runs preprocessing
        """
        start_date_time = datetime.now()

        ants_img = self.resample(self.input_data, self.reference_data)
        ants_img_mask = self.compute_mask(ants_img)
        ants_img = ants_img * ants_img_mask
        ants_img = self.compute_N4(ants_img, mask=ants_img_mask)
        vmin, vmax = np.min(ants_img.view()), np.max(ants_img.view())

        logger.info(f"Min max values: {vmin} {vmax}")

        ants_img, percentile_values = self.intensity_norm(ants_img)

        end_date_time = datetime.now()
        logger.info(
            f"Preprocessing complete, execution time:\
            {end_date_time - start_date_time} s"
        )

        return ants_img, percentile_values


def main(input_config: dict):
    """
    Main function to execute
    """

    mod = Preprocess(input_config)
    return mod.run()


if __name__ == "__main__":
    main()
