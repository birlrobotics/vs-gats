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

from model.vcoco_model import AGRNN
from datasets.vcoco.vsrl_eval import VCOCOeval
from datasets.vcoco_constants import VcocoConstants
from datasets.vcoco_dataset import VcocoDataset, collate_fn
from datasets import vcoco_metadata
import utils.io as io

def main(args):

    # use GPU if available else revert to CPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)

    # Load checkpoint and set up model
    try:
        # load checkpoint
        checkpoint = torch.load(args.pretrained, map_location=device)
        print('Checkpoint loaded!')

        # set up model and initialize it with uploaded checkpoint
        # ipdb.set_trace()
        if not args.exp_ver:
            args.exp_ver = args.pretrained.split("/")[-3]+"_"+args.pretrained.split("/")[-1].split("_")[-2]
        data_const = VcocoConstants(feat_type=checkpoint['feat_type'], exp_ver=args.exp_ver)
        model = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)

    io.mkdir_if_not_exists(data_const.result_dir)
    det_save_file = os.path.join(data_const.result_dir, 'detection_results.pkl')
    if not os.path.isfile(det_save_file) or args.rewrite:
        test_dataset = VcocoDataset(data_const=data_const, subset='vcoco_test')
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        # save detection result
        det_data_list = []
        # for global_id in tqdm(test_list): 
        for data in tqdm(test_dataloader):
            train_data = data
            global_id = train_data['global_id'][0]
            det_boxes = train_data['det_boxes'][0]
            roi_scores = train_data['roi_scores'][0]
            roi_labels = train_data['roi_labels'][0]
            node_num = train_data['node_num']
            features = train_data['features'] 
            spatial_feat = train_data['spatial_feat']
            word2vec = train_data['word2vec']

            # referencing
            features, spatial_feat, word2vec = features.to(device), spatial_feat.to(device), word2vec.to(device)
            outputs, attn, attn_lang = model(node_num, features, spatial_feat, word2vec, [roi_labels])    # !NOTE: it is important to set [roi_labels] 
            
            action_scores = nn.Sigmoid()(outputs)
            action_scores = action_scores.cpu().detach().numpy()
            attn = attn.cpu().detach().numpy()
            attn_lang = attn_lang.cpu().detach().numpy()

            h_idxs = np.where(roi_labels == 1)[0]
            # import ipdb; ipdb.set_trace()
            for h_idx in h_idxs:
                for i_idx in range(node_num[0]):
                    if i_idx == h_idx:
                        continue
                    # save hoi results in single image
                    single_result = {}
                    single_result['image_id'] = global_id
                    single_result['person_box'] = det_boxes[h_idx,:]
                    if h_idx > i_idx:
                        edge_idx = h_idx * (node_num[0] - 1) + i_idx
                    else:
                        edge_idx = h_idx * (node_num[0] - 1) + i_idx - 1
                    # score = roi_scores[h_idx] * roi_scores[i_idx] * action_score[edge_idx] * (attn[h_idx][i_idx-1]+attn_lang[h_idx][i_idx-1])
                    try:
                        score = roi_scores[h_idx] * roi_scores[i_idx] * action_scores[edge_idx]
                    except Exception as e:
                        import ipdb; ipdb.set_trace()
                    for action in vcoco_metadata.action_class_with_object:
                        if action == 'none':
                            continue
                        action_idx = vcoco_metadata.action_with_obj_index[action]
                        single_action_score = score[action_idx]
                        if action == 'cut_with' or action == 'eat_with' or action == 'hit_with':
                            action = action.split('_')[0]
                            role_name = 'instr'
                        else:
                            role_name = vcoco_metadata.action_roles[action][1]
                        action_role_key = '{}_{}'.format(action, role_name)
                        single_result[action_role_key] = np.append(det_boxes[i_idx,:], single_action_score)
                    
                    det_data_list.append(single_result)
        # save all detected results
        pickle.dump(det_data_list, open(det_save_file,'wb'))
    # evaluate
    vcocoeval = VCOCOeval(os.path.join(data_const.original_data_dir, 'data/vcoco/vcoco_test.json'),
                          os.path.join(data_const.original_data_dir, 'data/instances_vcoco_all_2014.json'),
                          os.path.join(data_const.original_data_dir, 'data/splits/vcoco_test.ids'))
    vcocoeval._do_eval(data_const, det_save_file, ovr_thresh=0.5)

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

    parser.add_argument('--exp_ver', '--e_v', type=str, default=None, 
                        help='the version of code, will create subdir in log/ && checkpoints/ ')

    parser.add_argument('--rewrite', '-r', action='store_true', default=False,
                        help='overwrite the detection file')

    args = parser.parse_args()
    # data_const = HicoConstants(feat_type=args.feat_type, exp_ver=args.exp_ver)
    # inferencing
    main(args)