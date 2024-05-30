# Template-based smartspim-ccf registration

This capsule is used to register SmartSPIM datasets to CCF Allen Atlas via SPIM template at 25 um.
`main.py` is compatible with the `aind-smartspim-pipeline`.
`main_register_dataset.py` is used to regsiter a stitched testing dataset.
It assumes that the fused image comes in OME-Zarr format with different multiscales.
At this point, we are using the 3rd multiscale from the original resolution for registration.

Workflow of template-based smartSPIM CCF registration:
1. preprocessing 
    - resample the raw image to isotropic to have the same resolution as the SPIM template.
    - masking, using Li thresholding to segment ROI
    - N4 bias correction
    - intensity normalization, using 2nd and 98th percentile normalization
2. register the resulting preprocessed image from step 1 to the SPIM template using ANTs (rigid + SyN). By default, the highest channel will be used for registration. 
3. register the resulting moved image from step 2 to the CCF Allen Atlas using the template-to-CCF transforms computed by Yoni.

## Usage
Please attach the data asset (both stitched and unstitched data you would like to run registration, i.e., `SmartSPIM_714635_2024-03-18_10-47-48` and `SmartSPIM_714635_2024-03-18_10-47-48_stitched_2024-03-28_04-43-39`) and update `subject_dir` line 121 in main.py, i.e., `subject_dir="SmartSPIM_714635_2024-03-18_10-47-48"`, then run 
```
conda activate ccf_reg
python main_register_dataset.py
```

## Output directory structure of registration
After running main.py given one testing dataset, a directory will be created with the following structure
```console
    /path/to/outputs/registration/
      ├── prep_*.nii.gz
      ├── prep_*.png
      ├── moved_rigid.nii.gz
      ├── moved_ls_to_template.nii.gz
      ├── moved_ls_to_ccf.nii.gz
      ├── moved_ccf_anno_to_ls.nii.gz
      ├── moved_*.png
      ├── reg_*.png
      └── ls_to_template_SyN*
```      
1. `prep_*.nii.gz`: the intermediate images in preprocessing steps, `prep_*.png` are the corresponding plots.
2. `moved_rigid.nii.gz`: the preprocessed brain image was aligned to the SPIM template using rigid registration.
3. `moved_ls_to_template.nii.gz`: the resulting image 2 was aligned to the SPIM template using SyN registration.
4. `moved_ls_to_ccf.nii.gz`: the resulting image 3 was aligned to the CCF using the template-to-CCF transforms computed by Yoni.
5. `moved_ccf_anno_to_ls.nii.gz`: the CCF annotation was aligned to the sample space.
6. `moved_*.png`: visualize the deformed images 2, 3, 4, 5.
7. `reg_*.png`: visualize the registration results for 2, 3, 4.
8. `ls_to_template_SyN*`: the transforms that align the preprocessed brain image to the SPIM template.

By default, the output file is for the channel-to-register (highest channel) if the file name does not contain the channel info.


## Running time

From Camilo's NFS 2024 abstract: 
```console
A typical dataset between 1 and 2 terabytes with 3 channels, 3.3 hours destriping and flat field, 0.2 hours transforms for stitching, 1.7 hours fusion to OMEZarr, 0.1 hours for registration to the 25 um template, 5-8 hours hours for cell detection and 0.2 hours for quantification. Total 10.5 hours without considering waiting times of the cluster and very labeled-dense datasets. Acquisition is about 3 hours per channel and data transfer from the microscope to the VAST and VAST to cloud about 20 mins.
10.5 + 3*(3) + ACQ_TO_VAST + 0.3 = 19.8
```
Di update on template-based smartSPIM-CCF registration:
```console
About 3 mins for preprocessing, 7 mins for registering to SPIM template, 9 seconds for registering to CCF. Total time: ~11 mins
```