import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, csv_path, split, label_dict, label_col='type', ignore=[]):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            split (pd.DataFrame): Train/val/test split. 
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int. 
            label_col (str, optional): Label column. Defaults to 'type'.
            ignore (list, optional): Ignored labels. Defaults to [].
        """        
        slide_data = pd.read_csv(csv_path)
        slide_data = self._df_prep(slide_data, label_dict, ignore, label_col)
        assert len(split) > 0, "Split should not be empty!"
        mask = slide_data['slide_id'].isin(split.tolist())
        self.slide_data = slide_data[mask].reset_index(drop=True)
        self.n_cls = len(set(label_dict.values()))
        self.slide_cls_ids = self._cls_ids_prep()
        self._print_info()

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return None

    def _print_info(self):
        print("Number of classes: {}".format(self.n_cls))
        print("Slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))

    def _cls_ids_prep(self):
        slide_cls_ids = [[] for i in range(self.n_cls)]
        for i in range(self.n_cls):
            slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        return slide_cls_ids

    def get_label(self, ids):
        return self.slide_data['label'][ids]

    @staticmethod
    def _df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data
    
class PatchFeatureDataset(BaseDataset):
    def __init__(self, data_path, low_mag, mid_mag, high_mag, **kwargs):
        """
        Args:
            data_path (str): Path to the data. 
            low_mag (str): Low magnifications. 
            mid_mag (str): Middle magnifications.
            high_mag (str): High magnifications.
        """        
        super(PatchFeatureDataset, self).__init__(**kwargs)
        self.data_path = data_path
        self.low_mag = low_mag
        self.mid_mag = mid_mag
        self.high_mag = high_mag
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        
        with h5py.File(os.path.join(self.data_path, '{}.h5'.format(slide_id)),'r') as hdf5_file:
            low_mag_feats = hdf5_file[f'{self.low_mag}_patches'][:]
            mid_mag_feats = hdf5_file[f'{self.mid_mag}_patches'][:]
            high_mag_feats = hdf5_file[f'{self.high_mag}_patches'][:]
        
        return torch.from_numpy(low_mag_feats), torch.from_numpy(mid_mag_feats), torch.from_numpy(high_mag_feats), label