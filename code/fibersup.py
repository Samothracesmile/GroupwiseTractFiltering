import os
import glob
import time
import copy
import random
import math

import numpy as np
import nibabel as nib
from nibabel import trackvis as tv
from nibabel.affines import apply_affine

from random import randint
from scipy.spatial import distance_matrix

# for zeropad
from skimage.util import pad


class Tract(object):
    def __init__(self, fname):
        self.fname = fname
        self.streams, self.hdr = tv.read(fname)
        self.streamlines = [np.nan_to_num(i[0]) for i in self.streams] #streamlines in list
        self.streamlines_untrimmed = [np.nan_to_num(i[0]) for i in self.streams] #streamlines in list

        self.affsums = [np.ones((len(j),),dtype=float) for j in self.streamlines] #affsum indicate the commonness distribution in terms of Ref streamlines

        self.cache_ref_dmc = [] # carry information


    def __len__(self):
        return len(self.streamlines)

    def streamline_label_update(self, Tha, lengthD = 60,lengthE = 20):
        streamlinesAA = []
        cacheAA = []

        untrimmed_streamlinesAA = []

        affAA = []

        for i in range(len(self.streamlines)):
        # for i in range(len(self.streamlines)):

            label = self.affsums[i] > Tha
            mask = firstbinarize(label)
            if len(label[mask]) > lengthD:
                if sum(label[mask]) >= len(label[mask]) - lengthE:

                    streamlinesAA = streamlinesAA + [self.streamlines[i][mask,:]]

                    if self.cache_ref_dmc:
                        cacheAA = cacheAA + [self.cache_ref_dmc[i][mask,:]]


                    affAA = affAA + [self.affsums[i][mask]]

                    untrimmed_streamlinesAA = untrimmed_streamlinesAA + [self.streamlines_untrimmed[i]]
        
        self.streamlines = streamlinesAA
        self.cache_ref_dmc = cacheAA

        self.affsums = affAA
        self.streamlines_untrimmed = untrimmed_streamlinesAA

    def convertTract2World(self, target_nifty):
        """
        target_tract: Tract object to be converted to the world space
        target_nifty: Nibabel nifty object provide affine information
        res_tract_fname: filename of trk file for counterpart in the world space 
        """
           
        affine = target_nifty.affine
        resolution = abs(affine[0][0])

        streamlines_in_world = [apply_affine(affine, streamline/resolution) for streamline in self.streamlines]

        return streamlines_in_world

    def convertTract2Image(self, target_nifty):
        """
        target_tract: Tract object to be converted to the world space
        target_nifty: Nibabel nifty object provide affine information
        res_tract_fname: filename of trk file for counterpart in the world space 
        """
           
        affine = target_nifty.affine
        resolution = affine[0][0]

        streamlines_in_image = [streamline/resolution for streamline in self.streamlines]

        return streamlines_in_image

    def subdownsampleTract(self, subsample_size, downsampling_rate = 1):
        """  
        subsample_size: number of streamlines in subsampled tract
        downsampling_rate: rate of downsampling of points in each streamline
        """
        streamline_length = len(self.streamlines)

        # avoid the large subsample_size
        subsample_size = min(streamline_length,subsample_size)

        random.seed(0)
        random_intlist = random.sample(range(streamline_length), subsample_size)
        subsample_tract_streamlines = [self.streamlines[i][::downsampling_rate,:] for i in random_intlist]    

        return subsample_tract_streamlines

    def subsampleTract(self, subsample_size):
        """  extracted_bundle
        subsample_size: number of streamlines in subsampled tract
        """
        streamline_length = len(self.streamlines)

        # avoid the large subsample_size
        subsample_size = min(streamline_length,subsample_size)

        random.seed(0)
        random_intlist = random.sample(range(streamline_length), subsample_size)
        subsample_tract_streamlines = [self.streamlines[i] for i in random_intlist]    

        return subsample_tract_streamlines    
    
    def downsampleTract(self, downsampling_rate = 1):
        """  
        downsampling_rate: rate of downsampling of points in each streamline
        """
        downsample_tract_streextracted_bundleamlines = [streamline[::downsampling_rate,:] for streamline in self.streamlines]    

        return downsample_tract_streextracted_bundleamlines    
    
######################################################### commonness measurement #########################################################        
def firstbinarize(AA):
    
    if sum(AA) < 0.5:
        return AA

    else:    
        forwardA = AA
        antiforwardA = AA[::-1]

        forwardi = 0
        antiforwardi = 0

        for i in range(int(len(forwardA))):  
            if forwardA[i] > 0:
                forwardi = i
                break

        for i in range(int(len(forwardA))):
            if antiforwardA[i] > 0:
                antiforwardi = i
                break

        label = np.zeros(len(AA),)    
        if antiforwardi == 0:
            label[max(0,forwardi):] = int(1)
        else:
            label[max(0,forwardi):-max(1,antiforwardi)] = int(1)
        # print(antiforwardi)
        return label > 0

####################################################OLD#########################
# def save_tract(tract_bundle, filename, tract_hdr=None):
#     '''
#     tract_bundle: Numpy array list
#     tract_hdr: .trk head file 
#     '''


#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))

#     for_save = [(streamline,None,None) for streamline in tract_bundle]

#     if tract_hdr is not None:
#         tv.write(filename, tuple(for_save), tract_hdr)
#     else:
#         tv.write(filename, tuple(for_save))

####################################################NEW#########################
def save_tract(tract_bundle, filename, scales=None, props=None, tract_hdr=None, ref_trk_file=None):
    '''
    tract_bundle: Numpy array list
    tract_hdr: .trk head file
    scales: Numpy array list for scales
    '''
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


    if scales is None:
        scales = [None]*len(tract_bundle)

    if props is None:
        props = [None]*len(tract_bundle)


    # use the header form ref trk
    if ref_trk_file is not None:
        tract_hdr = Tract(ref_trk_file).header


    for_save = [(streamline,scale,prop) for streamline, scale, prop in zip(tract_bundle, scales, props)]

    if tract_hdr is not None:
        if not len(for_save)==tract_hdr['n_count']:
            new_header = tract_hdr.copy()
            new_header['n_count'] = len(for_save)

            tv.write(filename, tuple(for_save), new_header)
        else:
            tv.write(filename, tuple(for_save), tract_hdr)

    else:
        tv.write(filename, tuple(for_save))

######################################################### commonness measurement #########################################################
def distMatrix(s1,s2):
    s1_norm = np.expand_dims(np.sum(s1**2,axis=1), axis=1)
    s2_norm = np.expand_dims(np.sum(s2**2,axis=1), axis=0)

    return np.sqrt(s1_norm + s2_norm - 2.0 * np.matmul(s1, np.transpose(s2)))


######################################################### commonness measurement #########################################################

def extract_local_window2D(image, location, window_width):
    assert np.all(image.shape - location), "Location error, check the location array"
    
    x0, y0 = location
    offset = math.floor(window_width/2)
    pad_width = offset + 1
    img_padded = pad(image, pad_width=pad_width, mode='constant')

    x = x0 + pad_width - offset # 0
    y = y0 + pad_width - offset # 0

#     return img_padded, img_padded[x:x+window_width,y:y+window_width]
    return img_padded[x:x+window_width,y:y+window_width]


def extract_local_window3D(image, location, window_width):
    assert np.all(image.shape - location), "Location error, check the location array"
    
    x0, y0, z0 = location
    offset = math.floor(window_width/2)
    pad_width = offset# + 1
    img_padded = pad(image, pad_width=pad_width, mode='constant')

    x = x0 + pad_width - offset # 0
    y = y0 + pad_width - offset # 0
    z = z0 + pad_width - offset # 0

#     return img_padded, img_padded[x:x+window_width, y:y+window_width, z:z+window_width]
    return img_padded[x:x+window_width, y:y+window_width, z:z+window_width]


def extract_locals_window3D(image, locations, window_width):

    offset = math.floor(window_width/2)
    pad_width = offset# + 1
    img_padded = pad(image, pad_width=pad_width, mode='constant')
    
    for location in locations:
        assert np.all(image.shape - location), "Location error, check the location array"
        x0, y0, z0 = location
        x = x0 + pad_width - offset # 0
        y = y0 + pad_width - offset # 0
        z = z0 + pad_width - offset # 0
        img_window = img_padded[x:x+window_width, y:y+window_width, z:z+window_width]
    #     return img_padded, img_padded[x:x+window_width, y:y+window_width, z:z+window_width]
        yield np.expand_dims(img_window, axis=0)

def flythrough_bundle(image, bundle, window_width):
    
    for streamline in bundle:
        streamline = np.round(streamline).astype(int) # Ceiling is also recommended 
        streamline = streamline.astype(int) # Ceiling is also recommended 
    
        # np.round(streamlines_in_image[1])
        # np.floor(streamlines_in_image[1])
        # np.ceil(streamlines_in_image[1])
        point_indexs = [point for point in streamline]
        yield np.vstack([point_window for point_window in extract_locals_window3D(image, point_indexs, window_width)])
