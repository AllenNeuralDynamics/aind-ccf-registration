"""
File for utilities
"""

import json
import logging
import multiprocessing
import os
import platform
import time
from cloudvolume import CloudVolume
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import psutil
import pydantic
from glob import glob
from aind_ccf_reg.configs import PathLike
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)


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


def read_json_from_pydantic(
    path: PathLike, pydantic_class
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

def get_channel_translations(
        translation_params: dict,
        channel_to_register: str,
) -> list:
    """
    Identifies all the imaging channels in a dataset

    Parameters
    ----------
    translation_params: dict
        list of channels from processing manifest
    channel_to_register: str
        channel that is being used for registration

    Returns
    -------
    list
        list of channels formated: Ex_*_Em_*

    """
    
    additional_channels = []
    if "excitation" in translation_params.keys():
        ex_wavelengths = translation_params["excitation"]
        em_wavelengths = translation_params["emmission"]
        
        for ex, em in zip(ex_wavelengths, em_wavelengths):
            channel = f"Ex_{ex}_Em_{em}"
            if channel != channel_to_register:
                additional_channels.append(f"Ex_{ex}_Em_{em}")
    else:
        for key, channel in translation_params.items():
            if channel != channel_to_register:
                additional_channels.append(channel)
            
    return additional_channels
        
def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for the CCF Atlas Registration step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata of ccf registration \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)

def rotate_image(img: np.array, in_mat: np.array, reverse: bool):
    """
    Rotates axes of a volume based on orientation matrix.
    
    Parameters
    ----------
    img: np.array
        Image volume to be rotated
    in_mat: np.array
        3x3 matrix with cols indicating order of input array and rows
        indicating location to rotate axes into
        
    Returns
    -------
    img_out: np.array
        Image after being rotated into new orientation
    out_mat: np.array
        axes correspondance after rotating array. Should always be an
        identity matrix
    reverse: bool
        if you are doing forward or reverse registration
    
    """


    if not reverse:
        in_mat = in_mat.T

    original, swapped = np.where(in_mat)
    img_out = np.moveaxis(img, original, swapped)

    out_mat = in_mat[:, swapped]
    for c, row in enumerate(in_mat):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
            out_mat[val, val] *= -1
            
    return img_out, out_mat

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
        The raw image in its aquired orientation
    params : dict
        The orientation information from processing_manifest.json
    orientations: dict
        The axis order of the CCF reference atals

    Returns
    -------
    img_out : np.array
        The raw image oriented to the CCF
    """

    orient_mat = np.zeros((3, 3))
    acronym = ["", "", ""]

    for k, vals in enumerate(params):
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        if direction in orientations.keys():
            ref_axis = orientations[direction]
            orient_mat[dim, ref_axis] = 1
            acronym[dim] = direction[0]
        else:
            direction_flip = "_".join(direction.split("_")[::-1])
            ref_axis = orientations[direction_flip]
            orient_mat[dim, ref_axis] = -1
            acronym[dim] = direction[0]

    # check because there was a bug that allowed for invalid spl orientation
    # all vals should be postitive so just taking absolute value of matrix
    if "".join(acronym) == "spl":
        orient_mat = abs(orient_mat)

    img_out, out_mat = rotate_image(img, orient_mat, False)

    return img_out, orient_mat, out_mat

class create_precomputed():

    def __init__(self, ng_params):
        self.regions = ng_params['regions']
        self.scaling = ng_params['scale_params']
        self.save_path = ng_params['save_path']

    def save_json(self, fpath: str, info: dict):
        """
        Saves information jsons for precomputed format

        Parameters
        ----------
        fpath: str
            full file path to where the data will be saved 
        info: dict
            data to be saved to file
        """

        path = f"{fpath}/info"

        with open(path, 'w') as fp:
            json.dump(info, fp, indent=2)
        
        return

    def create_segmentation_info(self):
        """
        Builds formating for additional info file for segmentation
        precomuted defining the segmentation regions from CCFv3 and
        save to json
        """
    
        json_data = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(k) for k in self.regions.keys()],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": [str(v) for k, v in  self.regions.items()]
                    }
                ]
            }
        }
        
        fpath = f"{self.save_path}/segment_properties"
        self.save_json(fpath, json_data)

        return


    def build_scales(self):
        """
        Creates the scaling information for segmentation precomputed
        info file

        Return
        ------
        scales: dict
            The resolution scales of the segmentation precomputed
            pyramid
        """
    
        scales = []
        for s in range(self.scaling['num_scales']):
            scale = {
                "chunk_sizes": [self.scaling['chunk_size']],
                "encoding": self.scaling['encoding'],
                "compressed_segmentation_block_size": self.scaling['compressed_block'],
                "key": "_".join(
                    [str(int(r * f**s)) for r, f in zip(self.scaling['res'], self.scaling['factors'])]
                ),
                "resolution": [
                    int(r * f**s) for r, f in zip(self.scaling['res'], self.scaling['factors'])
                ],
                "size": [
                    int(d // f**s) for d, f in zip(self.scaling['dims'], self.scaling['factors'])
                ]                                     
            }
            scales.append(scale)
    
        return scales

    def build_precomputed_info(self):
        """
        builds info dictionary for segmentation precomputed info file

        Returns
        -------
        info: dict
            information dictionary for creating info file

        """
        info = {
            "type": "segmentation",
            "segment_properties": "segment_properties",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": self.build_scales()
        }

        self.save_json(self.save_path, info)
    
        return info

    def volume_info(self, scale: int):
        """
        Builds information for each scale of precomputed parymid

        Returns
        -------
        info: Cloudvolume Object
            All the scaling information for an individual level
        """
    
        info = CloudVolume.create_new_info(
            num_channels = 1, 
            layer_type = 'segmentation', 
            data_type = 'uint32',
            encoding = self.scaling['encoding'],
            resolution = [int(r * f**scale) for r, f in zip(self.scaling['res'], self.scaling['factors'])],
            voxel_offset = [0, 0, 0],
            chunk_size = self.scaling['chunk_size'],
            volume_size = [int(d // f**scale) for d, f in zip(self.scaling['dims'], self.scaling['factors'])]
        )
    
        return info
    
    def create_segment_precomputed(self, img: np.array):
        """
        Creates segmentation precomputed pyramid and saves files

        Parameters
        ----------
        img: np.array
            The image that is being converted to a precomputed format
        """
    
        for scale in range(self.scaling['num_scales']):
        
            if scale == 0:
                curr_img = img
            else:
                factor = [1 / 2**scale for d in img.shape]
                curr_img = ndi.zoom(img, tuple(factor), order = 0)
        
            info = self.volume_info(scale)
            vol = CloudVolume(f"file://{self.save_path}", info=info, compress = False)
            vol[:, :, :] = curr_img.astype('uint32')
        
        return

    def cleanup_seg_files(self):

        files = glob(f"{self.save_path}/**/*.br", recursive = True)

        for file in files:
            new_file = file[:-3]
            os.rename(file, new_file)

        return

def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(
        time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        time_points[:min_len], memory_usages[:min_len], label="Memory Usage"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight"
    )


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
        
        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = os.environ.get("CO_MEMORY")
    co_memory = int(co_memory) if co_memory else None

    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")

    if co_memory:
        logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")

    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(
        f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}"
    )

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(
        psutil.cpu_percent(percpu=True, interval=1)
    ):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def save_string_to_txt(txt: str, filepath: str, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")
