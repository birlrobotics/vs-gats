import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import numpy as np
import utils.io as io 
from datasets.hico_constants import HicoConstants
from datasets import metadata

import sys

class HicoDataset(Dataset):
    '''
    Args:
        subset: ['train', 'val', 'train_val', 'test']
    '''
    def __init__(self, data_const=HicoConstants(), subset='train'):
        super(HicoDataset, self).__init__()
        self.data_const = data_const
        self.subset_ids = self._load_subset_ids(subset)
        self.sub_app_data = self._load_subset_app_data(subset)
        self.sub_spatial_data = self._load_subset_spatial_data(subset)
        self.word2vec = h5py.File(self.data_const.word2vec, 'r')

    def _load_subset_ids(self, subset):
        global_ids = io.load_json_object(self.data_const.split_ids_json)
        bad_det_ids = io.load_json_object(self.data_const.bad_faster_rcnn_det_ids)
        # skip bad instance detection image with 0-1 det
        # !NOTE: How to reduce the number of bad instance detection images
        subset_ids = [id for id in global_ids[subset] if id not in bad_det_ids['0']+bad_det_ids["1"]]
        return subset_ids

    def _load_subset_app_data(self, subset):
        print(f'Using {self.data_const.feat_type} feature...')
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.hico_trainval_data, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.hico_test_data, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def _load_subset_spatial_data(self, subset):
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.trainval_spatial_feat, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.test_spatial_feat, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def get_obj_one_hot(self,node_ids):
        num_cand = len(node_ids)
        obj_one_hot = np.zeros([num_cand,80])
        for i, node_id in enumerate(node_ids):
            obj_idx = int(node_id)-1
            obj_one_hot[i,obj_idx] = 1.0
        return obj_one_hot

    def get_word2vec(self,node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[metadata.coco_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    # def get_verb_one_hot(self,hoi_ids):
    #     num_cand = len(hoi_ids)
    #     verb_one_hot = np.zeros([num_cand,len(self.verb_to_id)])
    #     for i, hoi_id in enumerate(hoi_ids):
    #         verb_id = self.verb_to_id[self.hoi_dict[hoi_id]['verb']]
    #         verb_idx = int(verb_id)-1
    #         verb_one_hot[i,verb_idx] = 1.0
    #     return verb_one_hot

    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        global_id = self.subset_ids[idx]

        data = {}
        single_app_data = self.sub_app_data[global_id]
        single_spatial_data = self.sub_spatial_data[global_id]
        data['global_id'] = global_id
        data['img_name'] = global_id + '.jpg'
        data['det_boxes'] = single_app_data['boxes'][:]
        data['roi_labels'] = single_app_data['classes'][:]
        data['roi_scores'] = single_app_data['scores'][:]
        data['node_num'] = single_app_data['node_num'].value
        data['node_labels'] = single_app_data['node_labels'][:]
        data['features'] = single_app_data['feature'][:]
        data['spatial_feat'] = single_spatial_data[:]
        data['node_one_hot'] = self.get_obj_one_hot(data['roi_labels'])
        data['word2vec'] = self.get_word2vec(data['roi_labels'])
        # import ipdb; ipdb.set_trace()
        return data

# for DatasetLoader
def collate_fn(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_data = {}
    batch_data['global_id'] =[]
    batch_data['img_name'] = []
    batch_data['det_boxes'] = []
    batch_data['roi_labels'] = []
    batch_data['roi_scores'] = []
    batch_data['node_num'] = []
    batch_data['node_labels'] = []
    batch_data['features'] = []
    batch_data['spatial_feat'] = []
    # batch_data['node_one_hot'] = []
    batch_data['word2vec'] = []
    for data in batch:
        batch_data['global_id'].append(data['global_id'])
        batch_data['img_name'].append(data['img_name'])
        batch_data['det_boxes'].append(data['det_boxes'])
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['roi_scores'].append(data['roi_scores'])
        batch_data['node_num'].append(data['node_num'])
        batch_data['node_labels'].append(data['node_labels'])
        batch_data['features'].append(data['features'])
        batch_data['spatial_feat'].append(data['spatial_feat'])
        # batch_data['node_one_hot'].append(data['node_one_hot'])
        batch_data['word2vec'].append(data['word2vec'])

    # import ipdb; ipdb.set_trace()
    batch_data['node_labels'] = torch.FloatTensor(np.concatenate(batch_data['node_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))
    batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(batch_data['spatial_feat'], axis=0))
    # batch_data['node_one_hot'] = torch.FloatTensor(np.concatenate(batch_data['node_one_hot'], axis=0))
    batch_data['word2vec'] = torch.FloatTensor(np.concatenate(batch_data['word2vec'], axis=0))

    return batch_data