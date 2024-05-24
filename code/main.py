"""
Main used in code ocean to execute capsule
"""

import json
import logging
import multiprocessing
import os
import subprocess
from datetime import datetime
import glob

from aind_ccf_reg import register, utils
from natsort import natsorted
from aind_ccf_reg.configs import PathLike
from aind_ccf_reg.utils import create_folder


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going to be stored

    Returns
    -----------
    logging.Logger
        Created logger 
        pointing to the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    LOGS_FILE = f"{output_log_path}/register_process.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

#     logging.disable("DEBUG")
    logging.disable(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Execution datetime: {CURR_DATE_TIME}")

    return logger


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary

def execute_command_helper(command: str, print_command: bool = False) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    """
    Main function to register a dataset
    """
    
    """
    data_folder = os.path.abspath("../data/")
    processing_manifest_path = f"{data_folder}/processing_manifest.json"
    acquisition_path = f"{data_folder}/acquisition.json"
    processing_manifest_path = f"{data_folder}/smartspim_test_dataset/derivatives/processing_manifest.json"
    """

    #--------------------------- TODO ----------------------------#    
    
    subject_dir = "SmartSPIM_694513_2023-09-30_00-03-18_stitched_2024-01-11_10-15-23"
    subject_dir = "SmartSPIM_709391_2024-01-08_20-45-17_stitched_2024-01-11_15-48-31"
    subject_dir = "SmartSPIM_710625_2024-03-29_10-22-21_stitched_2024-03-30_22-18-10"

    # subject_dir = "SmartSPIM_685111_2023-09-28_18-19-10_stitched_2024-01-11_10-16-44"
    # subject_dir = "SmartSPIM_693196_2023-09-28_23-12-22_stitched_2024-01-11_10-23-15"
    # subject_dir = "SmartSPIM_693197_2023-09-29_05-18-50_stitched_2024-01-11_13-16-50"

    data_folder = os.path.abspath("../data/")
    processing_manifest_path = f"{data_folder}/processing_manifest_639.json" 
    acquisition_path = f"{data_folder}/{subject_dir}/acquisition.json"
    
    #-------------------------------------------------------------#    

    template_path = os.path.abspath("../data/smartspim_lca_template/smartspim_lca_template_25.nii.gz")
    ccf_reference_path = os.path.abspath("../data/allen_mouse_ccf/average_template/average_template_25.nii.gz") 
    template_to_ccf_transform_path = [
        os.path.abspath("../data/spim_template_to_ccf/syn_1Warp.nii.gz"),
        os.path.abspath("../data/spim_template_to_ccf/syn_0GenericAffine.mat")]
    
    print(f"template_to_ccf_transform_path: {template_to_ccf_transform_path}")
    ccf_annotation_to_template_moved_path = os.path.abspath("../data/ccf_annotation_to_template_moved.nii.gz")
    
    #-------------------------------------------------------------#    
    
    if not os.path.exists(processing_manifest_path):
        raise ValueError("Processing manifest path does not exist!")

    pipeline_config = read_json_as_dict(processing_manifest_path)
    pipeline_config = pipeline_config.get("pipeline_processing")

    if pipeline_config is None:
        raise ValueError("Please, provide a valid processing manifest")
        
    """
    # Setting parameters based on pipeline
    sorted_channels = natsorted(pipeline_config["registration"]["channels"])

    # Getting highest wavelenght as default for registration
    channel_to_register = sorted_channels[-1]
    """
    
    all_zarr = glob.glob(f'{data_folder}/{subject_dir}/image_tile_fusing/OMEZarr/*.zarr')
    all_zarr = sorted(all_zarr)
    all_zarr = [ zarr.split("/")[-1].replace(".zarr", "") for zarr in all_zarr]
    channel_to_register = all_zarr[-1]
    # lower_channels      = all_zarr[:-1]
    
    #-------------------------------------------------------------#    
    
    if not os.path.exists(acquisition_path):
        raise ValueError("Acquisition path does not exist!")

    acquisition_json = read_json_as_dict(acquisition_path)
    acquisition_orientation = acquisition_json.get("axes")
    print(f"acquisition_orientation: {acquisition_orientation}")

    if acquisition_orientation is None:
        raise ValueError(
            f"Please, provide a valid acquisition orientation, acquisition: {acquisition_json}"
        )

    #-------------------------------------------------------------#    
    
    dataset_id = subject_dir.split("_")[1]
    results_folder = f"../results/{dataset_id}_to_ccf_{channel_to_register}"
    create_folder(results_folder)
    
    logger = create_logger(output_log_path=results_folder)
    logger.info(f"channel_to_register: {channel_to_register}")
    logger.info(
        f"Processing manifest {pipeline_config} provided in path {processing_manifest_path}"
    )
    
    reg_folder = os.path.abspath(f"{results_folder}/registration")
    metadata_folder = os.path.abspath(f"{results_folder}/metadata")

    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info(f"{'='*40} SmartSPIM CCF Registration {'='*40}")

    example_input = {
#         "input_data": "../data/fused", # TODO
        "input_data": f"../data/{subject_dir}/image_tile_fusing/OMEZarr/",
        "input_channel": channel_to_register,
        "input_scale": pipeline_config["registration"]["input_scale"],
        "input_orientation": acquisition_orientation,
        "bucket_path": "aind-open-data",
        "template_path": template_path, # SPIM template
        "ccf_reference_path": ccf_reference_path,
        "template_to_ccf_transform_path": template_to_ccf_transform_path,
        "ccf_annotation_to_template_moved_path": ccf_annotation_to_template_moved_path,
        "reference_res": 25,
        "output_data": os.path.abspath(f"{results_folder}/OMEZarr"),
        "metadata_folder": metadata_folder,
        "code_url": "https://github.com/AllenNeuralDynamics/aind-ccf-registration",
        "reg_folder": reg_folder,
        "prep_params": {
            "rawdata_figpath": f"{reg_folder}/prep_zarr_img.jpg",
            "rawdata_path": f"{reg_folder}/prep_zarr_img.nii.gz",
            "resample_figpath": f"{reg_folder}/prep_resampled_zarr_img.jpg",
            "resample_path": f"{reg_folder}/prep_resampled_zarr_img.nii.gz",
            "mask_figpath": f"{reg_folder}/prep_mask.jpg",
            "mask_path": f"{reg_folder}/prep_mask.nii.gz",
            "n4bias_figpath": f"{reg_folder}/prep_n4bias.jpg",
            "n4bias_path": f"{reg_folder}/prep_n4bias.nii.gz",
            "img_diff_n4bias_figpath": f"{reg_folder}/prep_img_diff_n4bias.jpg",
            "img_diff_n4bias_path": f"{reg_folder}/prep_img_diff_n4bias.nii.gz",
            "percNorm_figpath": f"{reg_folder}/prep_percNorm.jpg",
            "percNorm_path": f"{reg_folder}/prep_percNorm.nii.gz",
            },
        "ants_params": {
            "spacing": (0.0144, 0.0144, 0.016),
            "unit": "millimetre",
            # "ccf_orientations": {
            #     "anterior_to_posterior": 0,
            #     "superior_to_inferior": 1,
            #     "left_to_right": 2,
            # }, 
            "template_orientations": {
                "anterior_to_posterior": 1,
                "superior_to_inferior": 2,
                "right_to_left": 0,
            }, 
            "rigid_path": f"{reg_folder}/moved_rigid.nii.gz",
            "moved_to_template_path": f"{reg_folder}/moved_ls_to_template.nii.gz",
            "moved_to_ccf_path": f"{reg_folder}/moved_ls_to_ccf.nii.gz",
            "ccf_anno_to_brain_path": f"{reg_folder}/moved_ccf_anno_to_ls.nii.gz",
        },
        "OMEZarr_params": {
            "clevel": 1,
            "compressor": "zstd",
            "chunks": (64, 64, 64),
        },
    }

    logger.info(f"Input parameters in CCF run: {example_input}")
    # flake8: noqa: F841
    image_path = register.main(example_input)

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_folder,
            "smartspim_ccf_registration",
        )


if __name__ == "__main__":
    main()
