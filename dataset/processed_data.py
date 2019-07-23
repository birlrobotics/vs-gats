import os
import argparse
import ipdb
from tqdm import tqdm

import numpy as np
import scipy.io as scio
import pickle

import metadata

def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))

def compute_area(box):
    return (box[2]-box[0])*(box[3]-box[1])

def get_node_index(classname, bbox, det_classes, det_boxes, node_num):
    bbox = np.array(bbox, dtype=np.float32)
    if classname == 'person':
        max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    else:
        max_iou = 0.35  
    max_iou_index = -1

    for i_node in range(node_num):
        # print(classname, metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]])
        if classname == metadata.hico_classes[det_classes[i_node]]:
            # check bbox overlap
            intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node, :]))
            iou = intersection_area/(compute_area(bbox)+compute_area(det_boxes[i_node, :])-intersection_area)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i_node
    return max_iou_index

def parse_data(args):

    assert os.path.exists(args.root), 'Please check the path to annotion file!'

    anno_data = scio.loadmat(args.root + '/anno_bbox.mat')
    print('Load original data successfully!')

    action_class_num = len(metadata.action_classes)
    list_action = anno_data['list_action']

    # parsing data
    for phase in ['bbox_train', 'bbox_test']:
        # if phase == 'bbox_train': continue
        sub_dir = phase.split('_')[-1] + '2015'
        if not os.path.exists('processed/'+sub_dir):
            os.makedirs('processed/'+sub_dir)

        data = anno_data[phase]
        for i_img in tqdm(range(data.shape[1])):
            # load detection data
            # i_img = 35672
            img_name = data['filename'][0,i_img][0]
            det_feat = np.load(os.path.join('detection_data', sub_dir, img_name.split(".")[0] + '_boxes_feature.npy'))
            det_class = np.load(os.path.join('detection_data', sub_dir, img_name.split(".")[0] + '_det_class.npy'))
            det_boxes = np.load(os.path.join('detection_data', sub_dir, img_name.split(".")[0] + '_det_boxes.npy'))
            det_scores = np.load(os.path.join('detection_data', sub_dir, img_name.split(".")[0] + '_det_scores.npy'))

            # sweep away the 'N/A' proposals
            keep_idx = []
            for idx in range(det_class.shape[0]):
                if not metadata.coco_classes[det_class[idx]] == 'N/A':
                    keep_idx.append(idx)
            det_feat = det_feat[keep_idx, :]
            det_boxes = det_boxes[keep_idx, :]
            det_class = det_class[keep_idx]
            det_scores = det_scores[keep_idx]

            # from coco label to hico label
            det_class = np.array(metadata.coco_to_hico)[det_class].astype(int)

            # calculate the number of nodes
            human_num = len(np.where(det_class == 50))
            obj_num = len(det_class) - human_num
            node_num = human_num + obj_num

            # parse the original data to get node labels
            node_labels = np.zeros((node_num, action_class_num))
            # ipdb.set_trace()
            for i_hoi in range(data['hoi'][0,i_img]['id'].shape[1]):
                try:
                    for j in range(data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x1'].shape[1]):
                        classname = 'person'
                        x1 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x1'][0, j][0, 0]
                        y1 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['y1'][0, j][0, 0]
                        x2 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x2'][0, j][0, 0]
                        y2 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['y2'][0, j][0, 0]
                        human_index = get_node_index(classname, [x1, y1, x2, y2], det_class, det_boxes, node_num)

                        hoi_id = data['hoi'][0, i_img]['id'][0, i_hoi][0, 0]
                        classname = list_action['nname'][hoi_id-1, 0][0]    # !NOTE: Need to subtract 1
                        x1 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['x1'][0, j][0, 0]
                        y1 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['y1'][0, j][0, 0]
                        x2 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['x2'][0, j][0, 0]
                        y2 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['y2'][0, j][0, 0]
                        obj_index = get_node_index(classname, [x1, y1, x2, y2], det_class, det_boxes, node_num)

                        action_id = metadata.hoi_to_action[hoi_id - 1]  # !NOTE: Need to subtract 1 
                        if human_index != -1 and obj_index != -1:
                            node_labels[human_index, action_id] = 1
                            node_labels[obj_index, action_id] = 1
                except IndexError:
                    pass

            # save precessed data
            instance = dict()
            # instance['human_num'] = human_num
            # instance['obj_num'] = obj_num
            instance['node_num'] = node_num
            instance['img_name'] = img_name
            instance['boxes'] = det_boxes
            instance['classes'] = det_class
            instance['scores'] = det_scores
            instance['node_labels'] = node_labels
            instance['feature'] = det_feat
            pickle.dump(instance, open(os.path.join(args.save_dir, sub_dir, '{}.p'.format(img_name.split(".")[0])), 'wb')) 

    print('Finished parsing datas!')    

if __name__ == "__main__":

    parse = argparse.ArgumentParser("Parse the HICO annotion data!!!")
    parse.add_argument("--root", '-r', type=str, default='hico',
                        help="where the anno_bbox.mat file is")
    parse.add_argument("--save_dir", type=str, default='processed',
                        help="where to save the processed data")
    args = parse.parse_args()

    # prese training data
    parse_data(args)


