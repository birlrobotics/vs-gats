# from __future__ import print_function
import os
import time
import h5py
import warnings
import argparse
import random

import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import datasets.vcoco.vsrl_eval
import datasets.vcoco.vsrl_utils as vu
from datasets import vcoco_metadata
from datasets.vcoco_constants import VcocoConstants
from utils.bbox_utils import compute_iou
from utils.vis_tool import vis_img_vcoco
import utils.io as io

matplotlib.use('TKAgg')

def get_node_index(bbox, det_boxes, index_list):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.3  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in index_list:
        # check bbox overlap
        iou = compute_iou(bbox, det_boxes[i_node, :])
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i_node
    return max_iou_index

def parse_data(data_const, args):
    # just focus on HOI samplers, remove those action with on objects
    action_class_num = len(vcoco_metadata.action_classes) - len(vcoco_metadata.action_no_obj)
    # no_action_index = vcoco_metadata.action_index['none']
    no_role_index = vcoco_metadata.role_index['none']
    # Load COCO annotations for V-COCO images
    coco = vu.load_coco()
    for subset in ["vcoco_train", "vcoco_test", "vcoco_val"]:
        # create file object to save the parsed data
        if not args.vis_result:
            print('{} data will be saved into {}/vcoco_data.hdf5 file'.format(subset.split("_")[1], subset))
            hdf5_file = os.path.join(data_const.proc_dir, subset, 'vcoco_data.hdf5')
            save_data = h5py.File(hdf5_file, 'w')
            # evaluate detection
            eval_det_file = os.path.join(data_const.proc_dir, subset, 'eval_det_result.json')
            gt_record = {n:0 for n in vcoco_metadata.action_class_with_object}
            det_record = gt_record.copy()

        # load selected data
        selected_det_data = h5py.File(os.path.join(data_const.proc_dir, subset, "selected_coco_cls_dets.hdf5"), 'r')

        # Load the VCOCO annotations for vcoco_train image set
        vcoco_all = vu.load_vcoco(subset)
        for x in vcoco_all:
            x = vu.attach_gt_boxes(x, coco)
            # record groundtruths
            if x['action_name'] in vcoco_metadata.action_class_with_object:
                if len(x['role_name']) == 2:
                    gt_record[x['action_name']] = sum(x['label'][:,0])
                else:
                    for i in range(x['label'].shape[0]):
                        if x['label'][i,0] == 1:
                            role_bbox = x['role_bbox'][i, :] * 1.
                            role_bbox = role_bbox.reshape((-1, 4))
                            for i_role in range(1, len(x['role_name'])):
                                if x['role_name'][i_role]=='instr' and (not np.isnan(role_bbox[i_role, :][0])):
                                    gt_record[x['action_name']+'_with'] +=1
                                    continue
                                if x['role_name'][i_role]=='obj' and (not np.isnan(role_bbox[i_role, :][0])):
                                    gt_record[x['action_name']] +=1                               
        # print(gt_record)
        image_ids = vcoco_all[0]['image_id'][:,0].astype(int).tolist()
        # all_results = list()
        unique_image_ids = list()
        for i_image, image_id in enumerate(image_ids):
            img_name = coco.loadImgs(ids=image_id)[0]['coco_url'].split('.org')[1][1:]
            # get image size
            img_gt = Image.open(os.path.join(data_const.original_image_dir, img_name)).convert('RGB')
            img_size = img_gt.size
            # load corresponding selected data for image_id 
            det_boxes = selected_det_data[str(image_id)]['boxes_scores_rpn_ids'][:,:4]
            det_scores = selected_det_data[str(image_id)]['boxes_scores_rpn_ids'][:,4]
            det_classes = selected_det_data[str(image_id)]['boxes_scores_rpn_ids'][:,-1].astype(int)
            det_features = selected_det_data[str(image_id)]['features']
            # calculate the number of nodes
            human_num = len(np.where(det_classes==1)[0])
            node_num = len(det_classes)
            obj_num = node_num - human_num
            labeled_edge_num = human_num * (node_num-1) 
            # labeled_edge_num = human_num * obj_num      # test: just consider h-o
            if image_id not in unique_image_ids:
                unique_image_ids.append(image_id)
                # construct empty edge labels
                edge_labels = np.zeros((labeled_edge_num, action_class_num))
                edge_roles = np.zeros((labeled_edge_num, 3))
                # edge_labels[:, no_action_index]=1    
                edge_roles[:, no_role_index] = 1
            else:
                if not args.vis_result:
                    edge_labels = save_data[str(image_id)]['edge_labels']
                    edge_roles = save_data[str(image_id)]['edge_roles']
                else:
                    continue
            # import ipdb; ipdb.set_trace()
            # Ground truth labels
            for x in vcoco_all:
                if x['label'][i_image,0] == 1:
                    if x['action_name'] in vcoco_metadata.action_no_obj:
                        continue
                    # role_bbox contain (agent,object/instr)
                    # if i_image == 16:
                    #     import ipdb; ipdb.set_trace()
                    role_bbox = x['role_bbox'][i_image, :] * 1.
                    role_bbox = role_bbox.reshape((-1, 4))
                    # match human box
                    bbox = role_bbox[0, :]
                    human_index = get_node_index(bbox, det_boxes, range(human_num))
                    if human_index == -1:
                        warnings.warn('human detection missing')
                        # print(img_name)
                        continue
                    assert human_index < human_num
                    # match object box
                    for i_role in range(1, len(x['role_name'])):
                        action_name = x['action_name']
                        if x['role_name'][i_role]=='instr' and (x['action_name'] == 'cut' or x['action_name'] == 'eat' or x['action_name'] == 'hit'):
                            action_index = vcoco_metadata.action_with_obj_index[x['action_name']+'_with']
                            action_name +='_with'
                            # import ipdb; ipdb.set_trace()
                            # print('testing')
                        else:
                            action_index = vcoco_metadata.action_with_obj_index[x['action_name']]
                        bbox = role_bbox[i_role, :]
                        if np.isnan(bbox[0]):
                            continue
                        if args.vis_result:
                            img_gt = vis_img_vcoco(img_gt, [role_bbox[0,:], role_bbox[i_role,:]], 1, raw_action=action_index, data_gt=True)
                        obj_index = get_node_index(bbox, det_boxes, range(node_num))    # !Note: Take the human into account
                        # obj_index = get_node_index(bbox, det_boxes, range(human_num, node_num))  # test
                        if obj_index == -1:
                            warnings.warn('object detection missing')
                            # print(img_name)
                            continue
                        if obj_index == human_index:
                            warnings.warn('human detection is the same to object detection')
                            # print(img_name)
                            continue
                        # match labels
                        # if human_index == 0:
                        #     edge_index = obj_index - 1
                        if human_index > obj_index:
                            edge_index = human_index * (node_num-1) + obj_index
                        else:
                            edge_index = human_index * (node_num-1) + obj_index - 1
                            # edge_index = human_index * obj_num + obj_index - human_num  #test
                        det_record[action_name] +=1
                        edge_labels[edge_index, action_index] = 1
                        # edge_labels[edge_index, no_action_index] = 0
                        edge_roles[edge_index, vcoco_metadata.role_index[x['role_name'][i_role]]] = 1
                        edge_roles[edge_index, no_role_index] = 0
                        
            # visualizing result instead of saving result
            if args.vis_result:
                # ipdb.set_trace()
                image_res = Image.open(os.path.join(data_const.original_image_dir, img_name)).convert('RGB')
                result = vis_img_vcoco(image_res, det_boxes, det_classes, det_scores, edge_labels, score_thresh=0.4)
                plt.figure(figsize=(100,100))
                plt.suptitle(img_name)
                plt.subplot(1,2,1)
                plt.imshow(np.array(img_gt))
                plt.title('all_ground_truth'+str(i_image))
                plt.subplot(1,2,2)
                plt.imshow(np.array(result))
                plt.title('selected_ground_truth')
                # plt.axis('off')
                plt.ion()
                plt.pause(1)
                plt.close()
            # save process data
            else:
                if str(image_id) not in save_data.keys():
                    # import ipdb; ipdb.set_trace()
                    save_data.create_group(str(image_id))
                    save_data[str(image_id)].create_dataset('img_name', data=np.fromstring(img_name, dtype=np.uint8).astype('float64'))
                    save_data[str(image_id)].create_dataset('img_size', data=img_size)
                    save_data[str(image_id)].create_dataset('boxes', data=det_boxes)
                    save_data[str(image_id)].create_dataset('classes', data=det_classes)
                    save_data[str(image_id)].create_dataset('scores', data=det_scores)
                    save_data[str(image_id)].create_dataset('feature', data=det_features)
                    save_data[str(image_id)].create_dataset('node_num', data=node_num)
                    save_data[str(image_id)].create_dataset('edge_labels', data=edge_labels)
                    save_data[str(image_id)].create_dataset('edge_roles', data=edge_roles)
                else:
                    save_data[str(image_id)]['edge_labels'][:] = edge_labels
                    save_data[str(image_id)]['edge_roles'][:] = edge_roles  
        if not args.vis_result:   
            save_data.close()      
            print("Finished parsing data!")   
        # eval object detection
        eval_single = {n:det_record[n]/gt_record[n] for n in vcoco_metadata.action_class_with_object}
        eval_all = sum(det_record.values()) / sum(gt_record.values())
        eval_det_result = {
            'gt': gt_record,
            'det': det_record,
            'eval_single': eval_single,
            'eval_all': eval_all
        }
        io.dump_json_object(eval_det_result, eval_det_file)

if __name__ == "__main__":

    parse = argparse.ArgumentParser("Parse the VCOCO annotion data!!!")
    parse.add_argument('--vis_result', '--v_r', action="store_true", default=False,
                        help='visualize the result or not')
    args = parse.parse_args()

    data_const = VcocoConstants()
    parse_data(data_const, args)
