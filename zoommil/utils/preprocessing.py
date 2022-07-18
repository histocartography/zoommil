from math import ceil, floor
from collections import OrderedDict
from typing import List, Optional, Tuple, Callable
from copy import deepcopy

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from histocartography.preprocessing.feature_extraction import GridDeepFeatureExtractor
from histocartography.pipeline import PipelineStep


class GridPatchExtractor(PipelineStep):
    def __init__(
        self,
        patch_size: int,
        stride: int = None,
        fill_value: int = 255,
        **kwargs,
    ) -> None:
        """
        Create a deep feature extractor.
        Args:
            patch_size (int): Desired size of patches.
            stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
            fill_value (int): Constant pixel value for image padding. Defaults to 255.
        """
        self.patch_size = patch_size
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride
        super().__init__(**kwargs)
        self.fill_value = fill_value

    def _process(self, input_image: np.ndarray) -> np.array:
        return self._extract_patches(input_image)

    def _extract_patches(self, input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches for a given RGB image.
        Args:
            input_image (np.ndarray): RGB input image.
        Returns:
            Tuple[np.array, np.array]: patches w/ dim=[n_patches, n_channels, patch_size, patch_size], indices w/ dim=[n_patches,2].
        """
        n_channels = input_image.shape[-1]
        patches = generate_patches(input_image,
                                   patch_size=self.patch_size,
                                   stride=self.stride)
        valid_indices = []
        valid_patches = []
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                valid_indices.append(np.array([row, col]))
                valid_patches.append(patches[row, col])

        valid_patches = np.array(valid_patches)
        valid_indices = np.array(valid_indices)
        indices = np.array([[row, col] for row in range(patches.shape[0]) for col in range(patches.shape[1])])
        patches = patches.reshape([-1, n_channels, self.patch_size, self.patch_size])
        return patches, indices


class MaskedGridPatchExtractor(GridPatchExtractor):
    def __init__(
        self,
        tissue_thresh: float = 0.1,
        **kwargs
    ) -> None:
        """
        Create a patch extractor that can process an image with a corresponding tissue mask.
        Args:
            tissue_thresh (float): Minimum fraction of tissue (vs background) for a patch to be considered as valid.
        """
        super().__init__(**kwargs)
        self.tissue_thresh = tissue_thresh

    def _process(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._extract_patches(input_image, mask)

    def _extract_patches(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate tissue mask and extract features of patches from a given RGB image.
        Record which patches are valid and which ones are not.
        Args:
            input_image (np.ndarray): RGB input image.
        Returns:
            Tuple[np.array, np.array]: patches w/ dim=[n_patches, n_channels, patch_size, patch_size], indices w/ dim=[n_patches,2].
        """
        mask = np.expand_dims(mask, axis=2)

        # load all the patches w/ shape = num_x X num_y x 3 x patch_size x patch_size 
        patches, mask_patches = generate_patches(
            input_image,
            patch_size=self.patch_size,
            stride=self.stride,
            mask=mask
        )

        valid_indices = []
        valid_patches = []
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                if self._validate_patch(mask_patches[row, col]):
                    valid_patches.append(patches[row, col])
                    valid_indices.append(str(row) + '_' + str(col))

        valid_patches = np.array(valid_patches)
        valid_indices = np.array(valid_indices)

        return valid_patches, valid_indices

    def _validate_patch(self, mask_patch: torch.Tensor) -> Tuple[List[bool], torch.Tensor]:
        """
        Record if patch is valid (sufficient area of tissue compared to background).
        Args:
            mask_patch (torch.Tensor): a mask patch.
        Returns:
            bool: Boolean filter for (in)valid patch
        """
        tissue_fraction = (mask_patch == 1).sum() / mask_patch.size
        if tissue_fraction >= self.tissue_thresh:
            return True
        return False


class MaskedGridDeepFeatureExtractor(GridDeepFeatureExtractor):
    def __init__(
        self,
        tissue_thresh: float = 0.1,
        seed: int = 1,
        **kwargs
    ) -> None:
        """
        Create a deep feature extractor that can process an image with a corresponding tissue mask.

        Args:
            tissue_thresh (float): Minimum fraction of tissue (vs background) for a patch to be considered as valid.
        """
        super().__init__(**kwargs)
        np.random.seed(seed)
        self.tissue_thresh = tissue_thresh
        self.transforms = None
        self.avg_pooler = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def _collate_patches(self, batch):
        """Patch collate function"""
        indices = [item[0] for item in batch]
        patches = [item[1] for item in batch]
        mask_patches = [item[2] for item in batch]
        patches = torch.stack(patches)
        mask_patches = torch.stack(mask_patches)
        return indices, patches, mask_patches

    def _process(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._extract_features(input_image, mask)

    def _extract_features(self, input_image: np.ndarray, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate tissue mask and extract features of patches from a given RGB image.
        Record which patches are valid and which ones are not.

        Args:
            input_image (np.ndarray): RGB input image.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Boolean index filter, patch features.
        """
        if self.downsample_factor != 1:
            input_image = self._downsample(input_image, self.downsample_factor)
            mask = self._downsample(mask, self.downsample_factor)
        mask = np.expand_dims(mask, axis=2)

        # create dataloader for image and corresponding mask patches
        masked_patch_dataset = MaskedGridPatchDataset(image=input_image,
                                                      mask=mask,
                                                      resize_size=self.resize_size,
                                                      patch_size=self.patch_size,
                                                      stride=self.stride,
                                                      mean=self.normalizer_mean,
                                                      std=self.normalizer_std)
        del input_image, mask
        patch_loader = DataLoader(masked_patch_dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self._collate_patches)

        # create dictionaries where the keys are the patch indices
        all_index_filter = OrderedDict(
            {(h, w): None for h in range(masked_patch_dataset.outshape[0])
                for w in range(masked_patch_dataset.outshape[1])}
        )
        all_features = deepcopy(all_index_filter)

        # extract features of all patches and record which patches are (in)valid
        indices = list(all_features.keys())
        offset = 0
        for _, img_patches, mask_patches in patch_loader:
            index_filter, features = self._validate_and_extract_features(img_patches, mask_patches)
            if len(img_patches) == 1:
                features = features.unsqueeze(dim=0)
            features = features.reshape(features.shape[0], 1024, 16, 16)
            features = self.avg_pooler(features).squeeze(dim=-1).squeeze(dim=-1).cpu().detach().numpy()
            for i in range(len(index_filter)):
                all_index_filter[indices[offset+i]] = index_filter[i]
                all_features[indices[offset+i]] = features[i]
            offset += len(index_filter)

        # convert to pandas dataframes to enable storing as .h5 files
        all_index_filter = pd.DataFrame(all_index_filter, index=['is_valid'])
        all_features = pd.DataFrame(np.transpose(np.stack(list(all_features.values()))),
                                    columns=list(all_features.keys()))

        return all_index_filter, all_features

    def _validate_and_extract_features(self, img_patches: torch.Tensor, mask_patches: torch.Tensor) -> Tuple[List[bool], torch.Tensor]:
        """
        Record which image patches are (in)valid.
        Extract features from the given image patches.

        Args:
            img_patches (torch.Tensor): Batch of image patches.
            mask_patches (torch.Tensor): Batch of mask patches.

        Returns:
            Tuple[List[bool], torch.Tensor]: Boolean filter for (in)valid patches, extracted patch features.
        """
        # record valid and invalid patches (sufficient area of tissue compared to background)
        index_filter = []
        for mask_p in mask_patches:
            tissue_fraction = (mask_p == 1).sum() / torch.numel(mask_p)
            if tissue_fraction >= self.tissue_thresh:
                index_filter.append(True)
            else:
                index_filter.append(False)

        # extract features of all patches unless all are invalid
        if any(index_filter):
            features = self.patch_feature_extractor(img_patches)
        else:
            features = torch.zeros(len(index_filter), 1024*16*16)
        return index_filter, features


class GridPatchDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        patch_size: int,
        resize_size: int,
        stride: int,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Create a dataset for a given image and extracted instance maps with desired patches
        of (size, size, 3).

        Args:
            image (np.ndarray): RGB input image.
            patch_size (int): Desired size of patches.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                               patches of size patch_size are provided to the network. Defaults to None.
            stride (int): Desired stride for patch extraction.
            mean (list[float], optional): Channel-wise mean for image normalization.
            std (list[float], optional): Channel-wise std for image normalization.
            transform (list[transforms], optional): List of transformations for input image.
        """
        super().__init__()
        basic_transforms = [transforms.ToPILImage()]
        self.resize_size = resize_size
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        if transform is not None:
            basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())
        if mean is not None and std is not None:
            basic_transforms.append(transforms.Normalize(mean, std))
        self.dataset_transform = transforms.Compose(basic_transforms)

        self.x_top_pad, self.x_bottom_pad = get_pad_size(image.shape[1], patch_size, stride)
        self.y_top_pad, self.y_bottom_pad = get_pad_size(image.shape[0], patch_size, stride)
        self.pad = torch.nn.ConstantPad2d((self.x_bottom_pad, self.x_top_pad, self.y_bottom_pad, self.y_top_pad), 255)
        self.image = torch.as_tensor(image)
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self._generate_patches(self.image)

    def _generate_patches(self, image):
        """Extract patches"""
        n_channels = image.shape[-1]
        patches = image.unfold(0, self.patch_size, self.stride).unfold(1, self.patch_size, self.stride)
        self.outshape = (patches.shape[0], patches.shape[1])
        patches = patches.reshape([-1, n_channels, self.patch_size, self.patch_size])
        return patches

    def __getitem__(self, index: int):
        """
        Loads an image for a given patch index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor]: Patch index, image as tensor.
        """
        patch = self.dataset_transform(self.patches[index].numpy().transpose([1, 2, 0]))
        return index, patch

    def __len__(self) -> int:
        return len(self.patches)


class MaskedGridPatchDataset(GridPatchDataset):
    def __init__(
        self,
        mask: np.ndarray,
        **kwargs
    ) -> None:
        """
        Create a dataset for a given image and mask, with extracted patches of (size, size, 3).

        Args:
            mask (np.ndarray): Binary mask.
        """
        super().__init__(**kwargs)

        self.mask_transform = None
        if self.resize_size is not None:
            basic_transforms = [transforms.ToPILImage(),
                                transforms.Resize(self.resize_size),
                                transforms.ToTensor()]
            self.mask_transform = transforms.Compose(basic_transforms)

        self.pad = torch.nn.ConstantPad2d((self.x_bottom_pad, self.x_top_pad, self.y_bottom_pad, self.y_top_pad), 0)
        self.mask = torch.as_tensor(mask)
        self.mask_patches = self._generate_patches(self.mask)

    def __getitem__(self, index: int):
        """
        Loads an image and corresponding mask patch for a given index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]: Patch index, image as tensor, mask as tensor.
        """
        image_patch = self.dataset_transform(self.patches[index].numpy().transpose([1, 2, 0]))
        if self.mask_transform is not None:
            # after resizing, the mask should still be binary and of type uint8
            mask_patch = self.mask_transform(255*self.mask_patches[index].numpy().transpose([1, 2, 0]))
            mask_patch = torch.round(mask_patch).type(torch.uint8)
        else:
            mask_patch = self.mask_patches[index]
        return index, image_patch, mask_patch


def get_pad_size(size: int, patch_size: int, stride: int) -> Tuple[int, int]:
    """Computes the necessary top and bottom padding size to evenly devide an input size into patches with a given stride
    Args:
        size (int): Size of input
        patch_size (int): Patch size
        stride (int): Stride
    Returns:
        Tuple[int, int]: Amount of top and bottom-pad
    """
    target = ceil((size - patch_size) / stride + 1)
    pad_size = ((target - 1) * stride + patch_size) - size
    top_pad = pad_size // 2
    bottom_pad = pad_size - top_pad
    return top_pad, bottom_pad


def pad_image_with_factor(input_image: np.ndarray, patch_size: int, factor: int = 1) -> np.ndarray:
    """
    Pad the input image such that the height and width is the multiple of factor * patch_size.
    Args:
        input_image (np.ndarray):   RGB input image.
        patch_size (int):           Patch size.
        factor (int):               Factor multiplied with the patch size.
    Returns:
        padded_image (np.ndarray): RGB padded image.
    """
    height, width = input_image.shape[0], input_image.shape[1]
    height_new, width_new = patch_size * factor * ceil(height/patch_size/factor), patch_size * factor * ceil(width/patch_size/factor)
    padding_top = floor((height_new - height)/2)
    padding_bottom = ceil((height_new - height)/2)
    padding_left = floor((width_new - width)/2)
    padding_right = ceil((width_new - width)/2)
    padded_image = np.copy(np.pad(input_image, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant', constant_values=255))

    return padded_image


def generate_patches(image, patch_size, stride, mask=None):
    """
    Extract patches on an image
    Args:
        image ([np.ndarray]): image to extract patches on 
        patch_size ([int]): extract patches of size patch_size x patch_size
        stride ([int]): patch stride
        mask ([np.ndarray], optional): extract same patches on associated mask. Defaults to None.
    Returns:
        [np.ndarray]: Extracted patches 
    """
    x_top_pad, x_bottom_pad = get_pad_size(image.shape[1], patch_size, stride)
    y_top_pad, y_bottom_pad = get_pad_size(image.shape[0], patch_size, stride)
    pad = torch.nn.ConstantPad2d((x_bottom_pad, x_top_pad, y_bottom_pad, y_top_pad), 255)
    image = pad(torch.as_tensor(np.array(image)).permute([2, 0, 1])).permute([1, 2, 0])

    patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride).detach().numpy()

    if mask is not None:
        pad = torch.nn.ConstantPad2d((x_bottom_pad, x_top_pad, y_bottom_pad, y_top_pad), 0)
        mask = pad(torch.as_tensor(np.array(mask)).permute([2, 0, 1])).permute([1, 2, 0])
        mask_patches = mask.unfold(0, patch_size, stride).unfold(1, patch_size, stride).detach().numpy()
        return patches, mask_patches 

    return patches


def get_image_at(image, magnification):
    """
    Get image at a specified magnification.
    Args:
        image (openslide.OpenSlide):    Whole-slide image opened with openslide.
        magnification (float):          Desired magnification.
    Returns:
        image_at (np.ndarray):          Image at given magnification.
    """

    # get image info
    down_factors = image.level_downsamples
    level_dims = image.level_dimensions
    if image.properties.get('aperio.AppMag') is not None:
        max_mag = int(image.properties['aperio.AppMag'])
        assert max_mag in [20, 40]
    else:
        print('WARNING: Assuming max. magnification is 40x!')
        max_mag = 40

    # get native magnifications
    native_mags = [max_mag / int(round(df)) for df in down_factors]

    # get image at the native magnification closest to the requested magnification
    if magnification in native_mags:
        down_level = native_mags.index(magnification)
    else:
        down_level = image.get_best_level_for_downsample(max_mag / magnification)
    print(f'Given magnification {magnification}, best level={down_level}, best mag={native_mags[down_level]}')
    image_at = image.read_region(location=(0, 0),
                                 level=down_level,
                                 size=level_dims[down_level]).convert('RGB')

    # downsample if necessary
    if native_mags[down_level] > magnification:
        w, h = image_at.size
        down_factor = int(native_mags[down_level] // magnification)
        print(f'Downsampling with factor {down_factor}')
        image_at = image_at.resize((w // down_factor, h // down_factor), Image.BILINEAR)

    # convert to np array -> h x w x c
    image_at = np.array(image_at)
    return image_at


def get_tissue_mask(image, mask_generator):
    # Color conversion
    img = image.copy()
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image_s = image_hsv[:, :, 1]

    # Otsu's thresholding
    _, raw_mask = cv2.threshold(image_s, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[raw_mask == 0] = 255

    # Extract mask
    refined_mask = mask_generator.process(image=img)
    return refined_mask