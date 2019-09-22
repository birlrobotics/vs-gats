from __future__ import print_function
import sys
import os
import ipdb
import pickle
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import dgl
import networkx as nx
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from model.model import AGRNN
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from datasets import metadata
import utils.io as io

def main(args):
    # use GPU if available else revert to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)

    # Load checkpoint and set up model
    try:
        # load checkpoint
        checkpoint = torch.load(args.pretrained, map_location=device)
        # in_feat, out_feat, hidden_size, action_num  = checkpoint['in_feat'], checkpoint['out_feat'],\
                                                        # checkpoint['hidden_size'], checkpoint['action_num']
        print('Checkpoint loaded!')

        # set up model and initialize it with uploaded checkpoint
        # ipdb.set_trace()
        data_const = HicoConstants(feat_type=checkpoint['feat_type'], exp_ver=args.exp_ver)
        model = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=True) #checkpoint['multi_head'])
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)
    
    print('Creating hdf5 file for predicting hoi dets ...')
    if not os.path.exists(data_const.result_dir):
        os.mkdir(data_const.result_dir)
    pred_hoi_dets_hdf5 = os.path.join(data_const.result_dir, 'pred_hoi_dets.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    # # print('Creating json file for predicted hoi dets ...')
    # hoi_box_score = {}
    # # prepare for data
    # dataset_list = io.load_json_object(data_const.split_ids_json)
    # test_list = dataset_list['test']
    # test_data = h5py.File(data_const.hico_test_data, 'r')
    test_dataset = HicoDataset(data_const=data_const, subset='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # for global_id in tqdm(test_list): 
    for data in tqdm(test_dataloader):
        # if global_id not in test_data.keys(): continue
        # train_data = test_data[global_id]
        train_data = data
        global_id = train_data['global_id'][0]
        img_name = train_data['img_name'][0]
        det_boxes = train_data['det_boxes'][0]
        roi_scores = train_data['roi_scores'][0]
        roi_labels = train_data['roi_labels'][0]
        node_num = train_data['node_num']
        # node_labels = train_data['node_labels']
        features = train_data['features'] 
        spatial_feat = train_data['spatial_feat']
        # node_one_hot = train_data['node_one_hot'] 
        word2vec = train_data['word2vec']
        
        # if node_num ==0 or node_num ==1: continue

        # referencing
        # features, spatial_feat, node_one_hot = features.to(device), spatial_feat.to(device), node_one_hot.to(device)
        features, spatial_feat, word2vec = features.to(device), spatial_feat.to(device), word2vec.to(device)
        outputs, atten = model(node_num, features, spatial_feat, word2vec, [roi_labels])    # !NOTE: it is important to set [roi_labels] 
        
        action_score = nn.Sigmoid()(outputs)
        action_score = action_score.cpu().detach().numpy()
        atten = atten.cpu().detach().numpy()
        # import ipdb; ipdb.set_trace()
        # save detection result
        # hoi_box_score[file_name.split('.')[0]] = {}
        pred_hois.create_group(global_id)
        det_data_dict = {}
        h_idxs = np.where(roi_labels == 1)[0]
        for h_idx in h_idxs:
            for i_idx in range(len(roi_labels)):
                if i_idx == h_idx:
                    continue
                if h_idx > i_idx:
                    score = roi_scores[h_idx] * roi_scores[i_idx] * (action_score[h_idx] + action_score[i_idx]) * atten[h_idx][i_idx]
                else:
                    score = roi_scores[h_idx] * roi_scores[i_idx] * (action_score[h_idx] + action_score[i_idx]) * atten[h_idx][i_idx-1]
                try:
                    hoi_ids = metadata.obj_hoi_index[roi_labels[i_idx]]
                except Exception as e:
                    ipdb.set_trace()
                for hoi_idx in range(hoi_ids[0]-1, hoi_ids[1]):
                    hoi_pair_score = np.concatenate((det_boxes[h_idx], det_boxes[i_idx], np.expand_dims(score[metadata.hoi_to_action[hoi_idx]], 0)), axis=0)
                    if str(hoi_idx+1).zfill(3) not in det_data_dict.keys():
                        det_data_dict[str(hoi_idx+1).zfill(3)] = hoi_pair_score[None,:]
                    else:
                        det_data_dict[str(hoi_idx+1).zfill(3)] = np.vstack((det_data_dict[str(hoi_idx+1).zfill(3)], hoi_pair_score[None,:]))
        for k, v in det_data_dict.items():
            pred_hois[global_id].create_dataset(k, data=v)
    # io.dump_json_object(hoi_box_score, os.path.join(result_path, 'pred_hoi_dets.json'))
    # pickle.dump(hoi_box_score, open(os.path.join(result_path, 'pred_hoi_dets.p'), 'wb'))
    pred_hois.close()

def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

if __name__ == "__main__":
    # set some arguments
    parser = argparse.ArgumentParser(description='Evaluate the model')

    parser.add_argument('--pretrained', '-p', type=str, default='checkpoints/v3_2048/epoch_train/checkpoint_300_epoch.pth', #default='checkpoints/v3_2048/epoch_train/checkpoint_300_epoch.pth',
                        help='Location of the checkpoint file: ./checkpoints/checkpoint_150_epoch.pth')

    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')

    # parser.add_argument('--feat_type', '--f_t', type=str, default='fc7', required=True, choices=['fc7', 'pool'],
    #                     help='if using graph head, here should be pool: default(fc7) ')

    parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', required=True,
                        help='the version of code, will create subdir in log/ && checkpoints/ ')

    args = parser.parse_args()
    # data_const = HicoConstants(feat_type=args.feat_type, exp_ver=args.exp_ver)
    # inferencing
    main(args)