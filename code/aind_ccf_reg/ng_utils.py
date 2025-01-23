#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:49:58 2024

@author: nicholas.lusk
"""

import os
import json
import subprocess
import numpy as np
import dask.array as da
from PIL import Image 

from pathlib import Path
from cloudvolume import CloudVolume
from argschema import ArgSchemaParser
from concurrent.futures import ProcessPoolExecutor

class Segmentation_precomputed(ArgSchemaParser):
    
    default_schema = 

    def save_json(file, path):
    
        with open(path, 'w') as fp:
            json.dump(file, fp)
        
        return

    def get_region_colors(tree, ccf):
    
        regions = np.unique(ccf)
    
        segment_color = {}
        region_dict = {}
        for region in regions:
            if region != 0:
                info = tree.get_structures_by_id([int(region)])
                hex_color = rgb_to_hex(info[0]['rgb_triplet'])
                
                segment_color[str(int(region))] = hex_color
                region_dict[str(int(region))] = info[0]['acronym']
    
        return segment_color, region_dict   

    def create_segmentation_info(regions):
    
        json_file = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(k) for k in regions.keys()],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": [str(v) for k, v in  regions.items()]
                    }
                ]
            }
        }

        return json_file

          
    def get_subsections(img, planes):
    
        ccf_sub = np.zeros((len(planes) + 1, img.shape[1], img.shape[2]), dtype = int)
    
        for c, plane in enumerate(planes):
            ccf_sub[c + 1, :, :] = img[int(plane), :, :]
            
        return ccf_sub

    def scale_ccf(img, output_size, pad = 5):
    
        img_out = np.zeros(output_size)
        
        for c in range(img.shape[0]):
            curr_img = img[c, :, :]
            padded_img = np.pad(curr_img, (pad, pad), mode = 'constant', constant_values = 0)
            padded_img = Image.fromarray(padded_img.astype(float))
            curr_out = padded_img.resize(output_size[-2:], resample = Image.NEAREST)
            img_out[c, :, :] = np.array(curr_out).T
        
        return img_out

    def build_scales(params):
    
        scales = []
        for s in range(params['num_scales']):
            scale = {
                "chunk_sizes": [params['chunk_size']],
                "encoding": params['encoding'],
                "compressed_segmentation_block_size": params['compressed_block'],
                "key": "_".join(
                    [str(r * f**s) for r, f in zip(params['res'], params['factors'])]
                ),
                "resolution": [int(r * f**s) for r, f in zip(params['res'], params['factors'])],
                "size": [int(d // f**s) for d, f in zip(params['dims'], params['factors'])]
                                                             
            }
            scales.append(scale)
    
        return scales

    def build_precomputed_info(params):
    
        info = {
            "type": "segmentation",
            "segment_properties": "segment_properties",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": build_scales(params['scale_params'])
        }
    
    return info

    def downsample_image(img, scale, factors):
    
        new_dims = [int(d // f**scale) for d, f in zip(img.shape, factors)]
    
        img_out = np.zeros(tuple(new_dims))
        for c in range(img.shape[0]):
            curr_img = img[c, :, :]
            curr_img = Image.fromarray(curr_img.astype(float))
            curr_out = curr_img.resize(new_dims[-2:], resample = Image.NEAREST)
            img_out[c, :, :] = np.array(curr_out).T
        
    return img_out
    

    def volume_info(params, scale):
    
        info = CloudVolume.create_new_info(
            num_channels = 1, 
            layer_type = 'segmentation', 
            data_type = 'uint32',
            encoding = params['encoding'],
            resolution = [int(r * f**scale) for r, f in zip(params['res'], params['factors'])],
            voxel_offset = [0, 0, 0],
            chunk_size = params['chunk_size'],
            volume_size = [int(d // f**scale) for d, f in zip(params['dims'], params['factors'])]
        )
    
        return info
    
    def save_volumes(img, params):
    
        for scale in range(params['scale_params']['num_scales']):
        
            if scale == 0:
                curr_img = img
            else:
                curr_img  = downsample_image(
                    img, 
                    scale, 
                    params['scale_params']['factors']
                )
            
            print(curr_img.shape)
        
            info = volume_info(params['scale_params'], scale)
            vol = CloudVolume(params['ng_path'], info=info, compress = 'br')
            
            vol[:, :, :] = curr_img.astype('uint32')
        
    
        return
    
    def run(self):
        return
    
def main(params: dict):
    
    return

if __name__ == "__main__":
    main(params)


