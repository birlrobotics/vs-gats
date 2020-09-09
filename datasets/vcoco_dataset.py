import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import numpy as np
import utils.io as io
from datasets import vcoco_metadata
from datasets.vcoco import vsrl_utils as vu
from datasets.vcoco_constants import VcocoConstants

import os
import sys
import random

class VcocoDataset(Dataset):
    '''
    Args:
        subset: ['vcoco_train', 'vcoco_val', 'vcoco_test', 'vcoco_trainval']
    '''
    data_sample_count = 0   # record how many times to process data sampling 

    def __init__(self, data_const=VcocoConstants(), subset='vcoco_train', data_aug=False, sampler=None):
        super(VcocoDataset, self).__init__()
        
        self.data_aug = data_aug
        self.data_const = data_const
        self.subset_ids = self._load_subset_ids(subset, sampler)
        self.sub_app_data = self._load_subset_app_data(subset)
        self.sub_spatial_data = self._load_subset_spatial_data(subset)
        self.word2vec = h5py.File(self.data_const.word2vec, 'r')

    def _load_subset_ids(self, subset, sampler):
        # import ipdb; ipdb.set_trace()
        vcoco = vu.load_vcoco(subset)
        subset_ids = list(set(vcoco[0]['image_id'][:,0].astype(int).tolist()))
        if sampler:
            # import ipdb; ipdb.set_trace()
            ''' when changing the model, use sub-dataset to quickly show if there is something wrong '''
            subset_ids = random.sample(subset_ids, int(len(subset_ids)*sampler))
        return subset_ids

    def _load_subset_app_data(self, subset):
        return h5py.File(os.path.join(self.data_const.proc_dir, subset, 'vcoco_data.hdf5'), 'r')

    def _load_subset_spatial_data(self, subset):
        return h5py.File(os.path.join(self.data_const.proc_dir, subset, 'spatial_feat.hdf5'), 'r')

    def _get_obj_one_hot(self,node_ids):
        num_cand = len(node_ids)
        obj_one_hot = np.zeros([num_cand,80])
        for i, node_id in enumerate(node_ids):
            obj_idx = int(node_id)-1
            obj_one_hot[i,obj_idx] = 1.0
        return obj_one_hot

    def _get_word2vec(self,node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[vcoco_metadata.coco_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    def _get_interactive_label(self, edge_label):
         
        interactive_label = np.zeros(edge_label.shape[0])  
        interactive_label = interactive_label[:, None]
        valid_idxs = list(set(np.where(edge_label==1)[0]))
        if len(valid_idxs) > 0:
            # import ipdb; ipdb.set_trace()
            interactive_label[valid_idxs,:] = 1
        return interactive_label

    def _data_sampler(self, data):
        # import ipdb; ipdb.set_trace()
        roi_labels = data['roi_labels']
        node_num = data['node_num']
        node_labels = data['node_labels']
        features = data['features']
        spatial_feat = data['spatial_feat']
        node_one_hot = data['node_one_hot']
        word2vec = data['word2vec']
        keep_inds = list(set(np.where(node_labels == 1)[0]))
        original_inds = np.arange(node_num)
        remain_inds = np.delete(original_inds, keep_inds, axis=0)
        random_select_num = 0 if remain_inds.shape[0]==0 else random.choice(np.arange(remain_inds.shape[0]))  #int(remain_inds.shape[0]-1) if int(remain_inds.shape[0]-1)>0 else 0
        random_select_inds = np.array(random.sample(remain_inds.tolist(), random_select_num), dtype=int) 
        choose_inds = sorted(np.hstack((keep_inds,random_select_inds)))
        # remove_inds = [x for x in original_inds if x not in choose_inds]
        if len(keep_inds)==0 or len(choose_inds)==1:
            return data    
        # re-construct the data 
        try:
            spatial_feat_inds = []
            for i in choose_inds:
                for j in choose_inds:
                    if i == j: 
                        continue
                    if j == 0:
                        ind = i * (node_num-1) + j
                    else:
                        ind = i * (node_num-1) + j - 1
                    spatial_feat_inds.append(ind)
            data['node_num'] = len(choose_inds)
            data['features'] = features[choose_inds,:]
            data['spatial_feat'] = spatial_feat[spatial_feat_inds,:]
            data['node_one_hot'] = node_one_hot[choose_inds,:]
            data['word2vec'] = word2vec[choose_inds,:]
            data['roi_labels'] = np.array([roi_labels[int(i)] for i in choose_inds])  # !NOTE, it is important to transfer list to np.array
            data['node_labels'] = node_labels[choose_inds, :]
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)
        VcocoDataset.data_sample_count+=1
        return data
        
    @staticmethod
    def displaycount():
        print("total times to process data sampling:", VcocoDataset.data_sample_count)

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
        single_app_data = self.sub_app_data[str(global_id)]
        single_spatial_data = self.sub_spatial_data[str(global_id)]
        data['global_id'] = global_id
        data['img_name'] = single_app_data['img_name']
        data['det_boxes'] = single_app_data['boxes'][:]
        data['roi_labels'] = single_app_data['classes'][:]
        data['roi_scores'] = single_app_data['scores'][:]
        data['node_num'] = single_app_data['node_num'].value
        # data['node_labels'] = single_app_data['node_labels'][:]
        data['edge_labels'] = single_app_data['edge_labels'][:]
        data['edge_num'] = data['edge_labels'].shape[0]
        data['features'] = single_app_data['feature'][:]
        data['spatial_feat'] = single_spatial_data[:]
        # data['node_one_hot'] = self._get_obj_one_hot(data['roi_labels'])
        data['word2vec'] = self._get_word2vec(data['roi_labels'])
        # data['interactive_label'] = self._get_interactive_label(data['edge_labels'])
        # import ipdb; ipdb.set_trace()
        if self.data_aug:
            thresh = random.random()
            if thresh > 0.5:
                data = self._data_sampler(data)
        return data

    # for inference
    def sample_date(self, global_id):
        data = {}
        single_app_data = self.sub_app_data[global_id]
        single_spatial_data = self.sub_spatial_data[global_id]
        data['global_id'] = global_id
        data['img_name'] = global_id + '.jpg'
        data['det_boxes'] = single_app_data['boxes'][:]
        data['roi_labels'] = single_app_data['classes'][:]
        data['roi_scores'] = single_app_data['scores'][:]
        data['node_num'] = single_app_data['node_num'].value
        # data['node_labels'] = single_app_data['node_labels'][:]
        data['edge_labels'] = single_app_data['edge_labels'][:]
        data['edge_num'] = data['edge_labels'].shape[0]
        data['features'] = single_app_data['feature'][:]
        data['spatial_feat'] = single_spatial_data[:]
        # data['node_one_hot'] = self._get_obj_one_hot(data['roi_labels'])
        data['word2vec'] = self._get_word2vec(data['roi_labels'])
        # data['interactive_label'] = self._get_interactive_label(data['edge_labels'])

        return data
# for DatasetLoader
def collate_fn(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_data = {}
    batch_data['global_id'] = []
    batch_data['img_name'] = []
    batch_data['det_boxes'] = []
    batch_data['roi_labels'] = []
    batch_data['roi_scores'] = []
    batch_data['node_num'] = []
    batch_data['edge_labels'] = []
    batch_data['edge_num'] = []
    # batch_data['node_labels'] = []
    batch_data['features'] = []
    batch_data['spatial_feat'] = []
    # batch_data['node_one_hot'] = []
    batch_data['word2vec'] = []
    # batch_data['interactive_label'] = []
    for data in batch:
        batch_data['global_id'].append(data['global_id'])
        batch_data['img_name'].append(data['img_name'])
        batch_data['det_boxes'].append(data['det_boxes'])
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['roi_scores'].append(data['roi_scores'])
        batch_data['node_num'].append(data['node_num'])
        # batch_data['node_labels'].append(data['node_labels'])
        batch_data['edge_labels'].append(data['edge_labels'])
        batch_data['edge_num'].append(data['edge_num'])
        batch_data['features'].append(data['features'])
        batch_data['spatial_feat'].append(data['spatial_feat'])
        # batch_data['node_one_hot'].append(data['node_one_hot'])
        batch_data['word2vec'].append(data['word2vec'])
        # batch_data['interactive_label'].append(data['interactive_label'])

    # import ipdb; ipdb.set_trace()
    # batch_data['node_labels'] = torch.FloatTensor(np.concatenate(batch_data['node_labels'], axis=0))
    batch_data['edge_labels'] = torch.FloatTensor(np.concatenate(batch_data['edge_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))
    batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(batch_data['spatial_feat'], axis=0))
    # batch_data['node_one_hot'] = torch.FloatTensor(np.concatenate(batch_data['node_one_hot'], axis=0))
    batch_data['word2vec'] = torch.FloatTensor(np.concatenate(batch_data['word2vec'], axis=0))
    # batch_data['interactive_label'] = torch.FloatTensor(np.concatenate(batch_data['interactive_label'], axis=0))

    return batch_data