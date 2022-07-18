from __future__ import print_function

import os
import time
import json
import random
import argparse
import pandas as pd
import numpy as np
import torch

from zoommil.config.config import Config
from zoommil.trainer import Trainer
from zoommil.dataset.dataset import PatchFeatureDataset

def datestr():
    now = time.gmtime()
    return '{:02}{:02}-{:02}{:02}{:02}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

def main(args):
    seed_torch(args.seed)

    # create results directory if necessary
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path) 
    
    all_splits = pd.read_csv(args.split_path)
    train_split = all_splits['train'].dropna().reset_index(drop=True)
    val_split = all_splits['val'].dropna().reset_index(drop=True)
    test_split = all_splits['test'].dropna().reset_index(drop=True)
    
    label_dict = {i: i for i in range(args.n_cls)}
    
    train_dataset = PatchFeatureDataset(csv_path=args.csv_path, split=train_split, label_dict=label_dict, label_col=args.label_col, 
                                        data_path=args.data_path, low_mag=args.low_mag, mid_mag=args.mid_mag, high_mag=args.high_mag)
    val_dataset = PatchFeatureDataset(csv_path=args.csv_path, split=val_split, label_dict=label_dict, label_col=args.label_col, 
                                      data_path=args.data_path, low_mag=args.low_mag, mid_mag=args.mid_mag, high_mag=args.high_mag)
    test_dataset = PatchFeatureDataset(csv_path=args.csv_path, split=test_split, label_dict=label_dict, label_col=args.label_col, 
                                       data_path=args.data_path, low_mag=args.low_mag, mid_mag=args.mid_mag, high_mag=args.high_mag)
    
    datasets = (train_dataset, val_dataset, test_dataset)
    trainer = Trainer(args, datasets)
    trainer.train()
    trainer.test()

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--config_path', type=str, default='./zoommil/config/sample_config.json',
                        help='path to configuration file (default: ./zoommil/config/sample_config.json)')
    args = parser.parse_args()

    # load config file
    with open(args.config_path, 'r') as ifile:
        config = Config(json.load(ifile))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = main(config)
    print("Done!")
