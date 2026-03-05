import os
import pydicom
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import random
import collections

class BrainS18Dataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        img_list = '/home/csantiago/Datasets/Radiology-CPTAC/full_file_location_CT50.txt'
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        #print("\n new", self.img_list)
        #custom_dir = r'D:\Datasets\manifest-xxn3N2Qq630907925598003437\'
        #self.root_dir = custom_dir
       
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.failed_shapes_file = 'CT_failed_shapes50.txt'
       

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.img_list)

   
    def __getitem__(self, idx):
        
        #idx = 520
        # get all images in directory
        dir_name = self.img_list[idx]
        #print(dir_name)
        #dir_name = r'D:\Datasets\manifest-xxn3N2Qq630907925598003437\TCGA-KIRC\TCGA-B8-A54G\08-19-2005-NA-CT RENAL Mass-87718\6.000000-Coronal Thin Mip-13360'
        img_names = sorted([os.path.join(dir_name, fname) for fname in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, fname)) and fname.endswith(".dcm")])
        
       
        # read all images and append pixel arrays to list
        pixel_arrays = []
        shapes = set()
        
        #index = 0
        
        for img_name in img_names:
            
            #print("\n INDEX:", index, img_name)
            
            assert os.path.isfile(img_name)
            img = pydicom.dcmread(img_name)
            
            #print("\n INDEX:", index, img_name, img)
            #print("\n INDEX:", index, img.pixel_array.shape)
            
            assert img is not None
            pixel_arrays.append(img.pixel_array)
            shapes.add(img.pixel_array.shape)

            #index = index +1 
        
        #print("\n Before file", "Shapes:", len(shapes), "Pixel_arrays:", len(pixel_arrays))
        
        if len(shapes) > 1:
            #print("\n opening file")
            with open(self.failed_shapes_file, "a") as f:
                f.write(dir_name + "\n")
            #print("\n end file")
            
        
        #print("\n Before if", len(shapes), len(pixel_arrays))
        
        if len(shapes) == 1:
            # All pixel arrays have the same shape
            #print("\n here1")
            np_pixel_arrays = np.array(pixel_arrays)
            #print("\n len1:", np_pixel_arrays.shape)

        elif len(shapes) == len(pixel_arrays):
            #print("\n here2")
            pixel_arrays = [pixel_arrays[0]]
            np_pixel_arrays = np.array(pixel_arrays)
            #print("\n len2:", np_pixel_arrays.shape)

        else:
            #print("\n here3")
            # determine the least frequent shape
            shape_counts = collections.Counter([pa.shape for pa in pixel_arrays])
            least_frequent_shape = min(shape_counts, key=shape_counts.get)

            # discard all pixel arrays with the least frequent shape
            pixel_arrays = [pa for pa in pixel_arrays if pa.shape != least_frequent_shape]
            np_pixel_arrays = np.array(pixel_arrays)
            #print("\n len3:", np_pixel_arrays.shape)
        
        #print("\n\n\n\n\n\n *********************THE END *********************** \n\n\n\n\n\n")

        # process pixel arrays
        processed_data = self.__testing_data_process__(np_pixel_arrays)

        # convert to tensor array
        tensor_array = self.__nii2tensorarray__(processed_data)
       

        return tensor_array
    
    
    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __testing_data_process__(self, data): 
        # crop data according net input size
    
        
        #print("\nShape of images:", data.shape)
    
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        #print("new", data.shape)
        return data