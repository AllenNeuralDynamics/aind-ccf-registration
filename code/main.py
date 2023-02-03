"""
Main used in code ocean to execute capsule
"""

import subprocess
import sys

from src.aind_ccf_reg import register


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
    image_path = register.main()
    bucket_path = "aind-open-data"

    output_folder = "/results"
    print(f"Bucket path: {bucket_path} - Output path: {output_folder}")
    # Copying output to bucket

    dataset_folder = str(sys.argv[2]).replace("/data/", "")
    channel_name = image_path.split("/")[-2].replace(".zarr", "")
    dataset_name = (
        dataset_folder + f"/processed/CCF_Atlas_Registration_{channel_name}"
    )
    s3_path = f"s3://{bucket_path}/{dataset_name}"

    for out in execute_command_helper(
        f"aws s3 mv --recursive {output_folder} {s3_path}"
    ):
        print(out)

    save_string_to_txt(
        f"Results of CCF registration saved in: {s3_path}",
        "/root/capsule/results/output_ccf.txt",
    )


if __name__ == "__main__":
    main()
