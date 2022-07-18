"""
Extract patches or patch features at different magnifications.
"""
import os
import argparse
import gc
import glob
from pathlib import Path
from math import ceil, floor
from collections import OrderedDict

import openslide
import numpy as np
import pandas as pd
import h5py

from histocartography.preprocessing.feature_extraction import GaussianTissueMask
from zoommil.utils.preprocessing import MaskedGridDeepFeatureExtractor, MaskedGridPatchExtractor
from zoommil.utils.preprocessing import pad_image_with_factor, get_image_at, get_tissue_mask


def run(out_path, in_path, mode, dataset, patch_size):

    # tissue mask and magnifications
    if dataset == 'CAMELYON16':
        tissue_threshold = 0.1
        magnifications = [10.0, 20.0]
        mask_generator = GaussianTissueMask(sigma=20, kernel_size=100, downsampling_factor=8)
        file_ext = ".svs"
    elif dataset == 'CRC':
        tissue_threshold = 0.5
        magnifications = [5.0, 10.0, 20.0]
        mask_generator = GaussianTissueMask(sigma=10, kernel_size=7, downsampling_factor=4)
    elif dataset == 'BRIGHT':
        tissue_threshold = 0.1
        magnifications = [1.25, 2.5, 5.0, 10.0]
        mask_generator = GaussianTissueMask(sigma=20, kernel_size=100, downsampling_factor=1)
        file_ext = ".svs"
    else:
        raise ValueError(f'Preprocessing for dataset "{dataset}" not implemented!')

    # get extractor
    if mode == 'features':
        extractor = MaskedGridDeepFeatureExtractor(architecture='resnet50',
                                                   patch_size=patch_size,
                                                   tissue_thresh=tissue_threshold,
                                                   downsample_factor=1,
                                                   extraction_layer='layer3',
                                                   batch_size=128,
                                                   num_workers=4,
                                                   seed=1)
    elif mode == 'patches':
        extractor = MaskedGridPatchExtractor(patch_size=patch_size, tissue_thresh=tissue_threshold)
    else:
        raise ValueError(f'Invalid mode "{mode}"!')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get file paths
    file_paths = glob.glob(os.path.join(in_path, f'*{file_ext}'))

    # extract patches or patch features for each image
    for fp in file_paths:
        filename = Path(fp).stem
        print(f'Open {fp}...')
        image = openslide.OpenSlide(fp)

        if mode == "features":
            print('Extract features...')
            valid_features = extract_features(image, extractor, mask_generator, magnifications, patch_size)

            print('Save as .h5 file...')
            for mag_key in valid_features.keys():
                coords_per_mag = []
                for key, val in valid_features[mag_key].items():
                    coords_per_mag.append(np.array([int(x) for x in key.split('_')]))
                feat_matrix = pd.DataFrame(valid_features[mag_key], columns=valid_features[mag_key].keys())
                feat_matrix = feat_matrix.to_numpy().transpose(-1, -2)
                with h5py.File(os.path.join(out_path, f'{filename}.h5'), 'a') as hf:
                    hf.create_dataset(f'{mag_key}_patches', data=feat_matrix)
                    hf.create_dataset(f'{mag_key}_coords', data=np.array(coords_per_mag))
                del feat_matrix
            del valid_features, image

        else:
            print('Extract patches...')
            valid_patches = extract_patches(image, extractor, mask_generator, magnifications, patch_size)

            print('Save as .h5 file...')
            for mag_key in valid_patches.keys():
                coords_per_mag = []
                patches_per_mag = []
                for key, val in valid_patches[mag_key].items():
                    patches_per_mag.append(val)
                    coords_per_mag.append(np.array([int(x) for x in key.split('_')]))
                with h5py.File(os.path.join(out_path, f'{filename}.h5'), 'a') as hf:
                    hf.create_dataset(mag_key + '_patches', data=np.array(patches_per_mag))  # dim = N x 3 x patch_size x patch_size 
                    hf.create_dataset(mag_key + '_coords', data=np.array(coords_per_mag))
                del patches_per_mag
                del coords_per_mag
            del valid_patches, image
        gc.collect()
            

def extract_features(image, extractor, mask_generator, magnifications, patch_size):

    # sort magnifications from lowest to highest
    magnifications.sort()
    features_all_mags = [[] for i in range(len(magnifications))]

    # assert that all factors between magnifications are 2
    assert set([magnifications[i+1]/magnifications[i] for i in range(len(magnifications)-1)]) == {2}

    # get image at the highest requested magnification
    high_mag_img = get_image_at(image, magnifications[-1])

    # pad such that image size is divisible by a multiple of the patch size
    high_mag_img = pad_image_with_factor(high_mag_img, patch_size=patch_size, factor=2**(len(magnifications)-1))
    high_mag_size = high_mag_img.shape

    # extract features at high magnification
    mask = np.ones(high_mag_img.shape[:-1])
    _, features = extractor.process(high_mag_img, mask)
    features_all_mags[-1] = features
    del high_mag_img, mask, features

    # extract features of the image at all other magnifications
    low_index = None
    for i_mag, mag in enumerate(magnifications[:-1]):
        # read image at given magnification and pad accordingly
        k = int(magnifications[-1] // mag)
        image_at_mag = get_image_at(image, mag)
        size = image_at_mag.shape
        top_pad = floor((high_mag_size[0]/k - size[0])/2)
        bottom_pad = ceil((high_mag_size[0]/k - size[0])/2)
        left_pad = floor((high_mag_size[1]/k - size[1])/2)
        right_pad = ceil((high_mag_size[1]/k - size[1])/2)
        image_at_mag = np.pad(image_at_mag, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)),
                                mode='constant', constant_values=255)

        if i_mag == 0:
            print('Get tissue mask...', flush=True)
            mask = get_tissue_mask(image_at_mag, mask_generator)
        else:
            # all the patches are artifically set to valid and removed later based on low mag mask
            mask = np.ones(image_at_mag.shape[:-1])
        
        # extract all features
        print('Extract features...', flush=True)
        index, features = extractor.process(image_at_mag, mask)

        # if no patch contains more tissue than the threshold, use all patches
        if not any([is_valid[0] for coords, is_valid in index.iteritems()]):
            print('Mask too restrictive, extracting all patches...', flush=True)
            index, features = extractor.process(image_at_mag, np.ones(mask.shape))

        del image_at_mag, mask
        if i_mag == 0:
            low_index = index
        features_all_mags[i_mag] = features
        del features

    # loop over valid patches at lowest magnifcation and recursively get all sub-patch features from higher magnifications
    valid_features = OrderedDict({f'{m}x': OrderedDict() for m in magnifications})
    for coords, is_valid in low_index.iteritems():
        idx_h, idx_w = coords
        prefix = f'{idx_h}_{idx_w}'
        if is_valid[0]:
            valid_features[f'{magnifications[0]}x'][prefix] = np.array(features_all_mags[0][coords])
            valid_features = get_valid_features(valid_features, features_all_mags, 1, magnifications, idx_h, idx_w, prefix)
    return valid_features


def extract_patches(image, extractor, mask_generator, magnifications, patch_size):

    # sort requested magnifications from lowest to highest
    magnifications.sort()

    # assert that all factors between magnifications are 2
    assert set([magnifications[i+1]/magnifications[i] for i in range(len(magnifications)-1)]) == {2}

    # get image at the highest requested magnification
    high_mag_img = get_image_at(image, magnifications[-1])

    # pad such that image size is divisible by a multiple of the patch size
    high_mag_img = pad_image_with_factor(high_mag_img, patch_size=patch_size, factor=2**(len(magnifications)-1))
    high_mag_size = high_mag_img.shape

    # extract patches of the image at all magnifications
    patches_all_mags = []
    indices_all_mags = []
    for i_mag, mag in enumerate(magnifications):
        if i_mag == len(magnifications)-1:
            # highest magnification image has already been extracted
            image_at_mag = high_mag_img
        else:
            # read image at given magnification and pad accordingly
            k = int(magnifications[-1] // mag)
            image_at_mag = get_image_at(image, mag)
            size = image_at_mag.shape           
            top_pad = floor((high_mag_size[0]/k - size[0])/2)
            bottom_pad = ceil((high_mag_size[0]/k - size[0])/2)
            left_pad = floor((high_mag_size[1]/k - size[1])/2)
            right_pad = ceil((high_mag_size[1]/k - size[1])/2)
            image_at_mag = np.pad(image_at_mag, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)),
                                  mode='constant', constant_values=255)

        if i_mag == 0:
            mask = get_tissue_mask(image_at_mag, mask_generator)
        else:
            # all the patches are artifically set to valid and removed later based on low mag mask.
            mask = np.ones(image_at_mag.shape[:-1])
    
        # extract all patches and indices 
        patches, indices = extractor.process(image_at_mag, mask)  # provide mask only at lowest magnification...

        # if no patch contains more tissue than the threshold, use all patches
        if len(indices) == 0:
            print('Mask too restrictive, extracting all patches...')
            patches, indices = extractor.process(image_at_mag, np.ones(mask.shape))

        patches_all_mags.append(patches)
        indices_all_mags.append(indices)
        del image_at_mag, mask
        del patches
    del high_mag_img

    # loop over valid patches at lowest magnifcation and recursively get all sub-patch patches from higher magnifications
    valid_patches = OrderedDict({f'{m}x': OrderedDict() for m in magnifications})
    for coords, patch in zip(indices_all_mags[0], patches_all_mags[0]):
        idx_h, idx_w = coords.split('_')
        idx_h = int(idx_h)
        idx_w = int(idx_w)
        prefix = coords
        valid_patches[f'{magnifications[0]}x'][prefix] = np.array(patch)
        if len(magnifications) > 1:
            valid_patches = get_valid_patches(valid_patches, patches_all_mags, indices_all_mags, 1, magnifications, idx_h, idx_w, prefix)
    
    return valid_patches


def get_valid_features(valid_features, all_features, i_mag, magnifications, idx_h, idx_w, prefix):
    """
    Recursively get all sub-patch features from higher magnifications.
    """

    mag_key = f'{magnifications[i_mag]}x'
    valid_features[mag_key][prefix + '_0_0'] = np.array(all_features[i_mag][(idx_h*2, idx_w*2)])
    valid_features[mag_key][prefix + '_1_0'] = np.array(all_features[i_mag][(idx_h*2+1, idx_w*2)])
    valid_features[mag_key][prefix + '_0_1'] = np.array(all_features[i_mag][(idx_h*2, idx_w*2+1)])
    valid_features[mag_key][prefix + '_1_1'] = np.array(all_features[i_mag][(idx_h*2+1, idx_w*2+1)])
    
    if i_mag != len(magnifications)-1:
        valid_features = get_valid_features(valid_features, all_features, i_mag+1, magnifications, idx_h*2, idx_w*2, prefix + '_0_0')
        valid_features = get_valid_features(valid_features, all_features, i_mag+1, magnifications, idx_h*2+1, idx_w*2, prefix + '_1_0')
        valid_features = get_valid_features(valid_features, all_features, i_mag+1, magnifications, idx_h*2, idx_w*2+1, prefix + '_0_1')
        valid_features = get_valid_features(valid_features, all_features, i_mag+1, magnifications, idx_h*2+1, idx_w*2+1, prefix + '_1_1')

    return valid_features


def get_valid_patches(valid_patches, all_patches, all_coords, i_mag, magnifications, idx_h, idx_w, prefix):
    """
    Recursively get all sub-patches from higher magnifications.
    """

    mag_key = f'{magnifications[i_mag]}x'

    idx = np.where(all_coords[i_mag] == (str(idx_h*2) + '_' + str(idx_w*2)))
    valid_patches[mag_key][prefix + '_0_0'] = np.squeeze(np.array(all_patches[i_mag][idx]))
    idx = np.where(all_coords[i_mag] == (str(idx_h*2+1) + '_' + str(idx_w*2))) 
    valid_patches[mag_key][prefix + '_1_0'] = np.squeeze(np.array(all_patches[i_mag][idx]))
    idx = np.where(all_coords[i_mag] == (str(idx_h*2) + '_' + str(idx_w*2+1)))
    valid_patches[mag_key][prefix + '_0_1'] = np.squeeze(np.array(all_patches[i_mag][idx]))
    idx = np.where(all_coords[i_mag] == (str(idx_h*2+1) + '_' + str(idx_w*2+1)))
    valid_patches[mag_key][prefix + '_1_1'] = np.squeeze(np.array(all_patches[i_mag][idx]))

    if i_mag != len(magnifications)-1:
        valid_patches = get_valid_patches(valid_patches, all_patches, all_coords, i_mag+1, magnifications, idx_h*2, idx_w*2, prefix + '_0_0')
        valid_patches = get_valid_patches(valid_patches, all_patches, all_coords, i_mag+1, magnifications, idx_h*2+1, idx_w*2, prefix + '_1_0')
        valid_patches = get_valid_patches(valid_patches, all_patches, all_coords, i_mag+1, magnifications, idx_h*2, idx_w*2+1, prefix + '_0_1')
        valid_patches = get_valid_patches(valid_patches, all_patches, all_coords, i_mag+1, magnifications, idx_h*2+1, idx_w*2+1, prefix + '_1_1')

    return valid_patches


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Configurations for Patch Feature Extraction')
    parser.add_argument('--out_path', default='/path/to/preprocessed/h5_files')
    parser.add_argument('--in_path', default='/path/to/wsis', help='path to whole-slide images')
    parser.add_argument('--mode', choices=['features', 'patches'], default='features',
                        help='extraction mode ("features" or "patches")')
    parser.add_argument('--dataset', choices=['BRIGHT', 'CAMELYON16', 'CRC'], default='BRIGHT',
                        help='dataset name ("BRIGHT", "CAMELYON16", or "CRC")')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size (default: 256)')
    args = parser.parse_args()

    run(out_path=args.out_path,
        in_path=args.in_path,
        mode=args.mode,
        dataset=args.dataset,
        patch_size=args.patch_size)
    print('Done!')
