
from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Str , Int, Nested
import random
import numpy as np
import ants
from dask.array import from_zarr
import s3fs
import zarr
import tifffile
from skimage import io


example_input = {
    "input_data": "s3://aind-open-data/SmartSPIM_621362_2022-10-14_14-08-48_2022_10_18_21_11_25_stitched/OMEZarr/Ex_639_Em_660.zarr/3/",
    "reference": "/Users/sharmishtaas/Desktop/expts/brainreg_jack/templates/allen_mouse_25um_v1.2/reference.tiff",
    "reference_res": 25,
    "output_data": "outfile.tiff",
    "downsampled_file": "downsampled.tiff",
    "downsampled16bit_file": "downsampled_16.tiff"

}


class RegSchema(ArgSchema):
    """
    Schema format for Registration.
    """
    input_data = Str(metadata={"required":True, "description":"Input data"})
    reference = Str(metadata={"required":True, "description":"Reference image"})
    output_data = Str(metadata={"required":True, "description":"Output file"})
    downsampled_file = Str(metadata={"required":True, "description":"Downsampled file"})
    downsampled16bit_file = Str(metadata={"required":True, "description":"Downsampled 16bit file"})
    reference_res = Int(metadata={"required":True, "description":"Voxel Resolution of reference in microns"})

class Register (ArgSchemaParser):
    """
    Class to Register lightsheet data to CCF atlas
    """
    default_schema = RegSchema

    def run(self):

        #read input data (lazy loading)
        print("Going to read zarr")
        if "s3:/" in self.args['input_data']:
            print("S3 data")
            s3 = s3fs.S3FileSystem(anon=True)
            store = s3fs.S3Map(root=self.args['input_data'], s3=s3, check=False)
            dask_array = zarr.open(store=store, mode='r')
            print(dask_array.shape)
            img_array = np.asarray(dask_array)
            img_array = np.squeeze(img_array)
        elif "gs:/" in self.args['input_data']:
            print("GCP data")
            dask_array = from_zarr(self.args['input_data'])
            print(dask_array.shape)
            img_array = np.asarray(dask_array)
            img_array = np.squeeze(img_array)[0]
        else:
            print("cannot recognize")
            exit(0)
        
        #get data orientation
        img_array = img_array.astype(np.double)
        img_array = np.swapaxes(img_array, 0, 2)
        img_array = np.swapaxes(img_array, 1,2)
        img_array = np.flip (img_array, 2)

        #convert input data to tiff into reference voxel resolution
        ants_img= ants.from_numpy(img_array, spacing=(14.4, 14.4, 16) )
        new_spacing = (25,25,25)
        fillin = ants.resample_image(ants_img,new_spacing,False,1)
        print("Size of resampled image: ", fillin.shape)
        ants.image_write(fillin,self.args['downsampled_file'])

        #convert data to uint16
        im = io.imread(self.args['downsampled_file']).astype(np.uint16)
        tifffile.imwrite(self.args['downsampled16bit_file'],im)


        #read images
        print("Reading reference image")
        img1 = ants.image_read(self.args['reference'])
        img2 = ants.image_read(self.args['downsampled16bit_file'])
        #img2 = ants.image_read('/Users/sharmishtaas/Desktop/expts/brainreg_jack/niftifreg/Rbp4_620795/488/dir1-allen25/output/downsampled.tiff')
        
        #register with ants
        reg12 = ants.registration( img1, img2, 'SyN', reg_iterations = [100,10,0] )

        #output
        ants.image_write(reg12['warpedmovout'],self.args['output_data'])

if __name__ == '__main__':
    mod = Register(example_input)
    mod.run()