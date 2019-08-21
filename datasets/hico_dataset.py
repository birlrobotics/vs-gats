import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import numpy as np
import utils.io as io 
from datasets.hico_constants import HicoConstants

import sys

class HicoDataset(Dataset):
    '''
    Args:
        subset: ['train', 'val', 'train_val', 'test']
    '''
    def __init__(self, data_const=HicoConstants(), subset='train',):
        super(HicoDataset, self).__init__()
        self.data_const = data_const
        self.subset_ids = self._load_subset_ids(subset)
        self.sub_app_data = self._load_subset_app_data(subset)

    def _load_subset_ids(self, subset):
        global_ids = io.load_json_object(self.data_const.split_ids_json)
        bad_det_ids = io.load_json_object(self.data_const.bad_faster_rcnn_det_ids)
        # skip bad instance detection image with 0-1 det
        # !NOTE: How to reduce the number of bad instance detection images
        subset_ids = [id for id in global_ids[subset] if id not in bad_det_ids['0']+bad_det_ids["1"]]
        return subset_ids

    def _load_subset_app_data(self, subset):
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.hico_trainval_data, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.hico_test_data, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        global_id = self.subset_ids[idx]

        data = {}
        single_data = self.sub_app_data[global_id]
        data['img_name'] = global_id + '.jpg'
        data['det_boxes'] = single_data['boxes'][:]
        data['roi_labels'] = single_data['classes'][:]
        data['roi_scores'] = single_data['scores'][:]
        data['node_num'] = single_data['node_num'].value
        data['node_labels'] = single_data['node_labels'][:]
        data['features'] = single_data['feature'][:]

        return data

def collate_fn(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_data = {}
    batch_data['img_name'] = []
    batch_data['det_boxes'] = []
    batch_data['roi_labels'] = []
    batch_data['roi_scores'] = []
    batch_data['node_num'] = []
    batch_data['node_labels'] = []
    batch_data['features'] = []
    for data in batch:
        batch_data['img_name'].append(data['img_name'])
        batch_data['det_boxes'].append(data['det_boxes'])
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['roi_scores'].append(data['roi_scores'])
        batch_data['node_num'].append(data['node_num'])
        batch_data['node_labels'].append(data['node_labels'])
        batch_data['features'].append(data['features'])

    # import ipdb; ipdb.set_trace()
    batch_data['node_labels'] = torch.FloatTensor(np.concatenate(batch_data['node_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))

    return batch_data