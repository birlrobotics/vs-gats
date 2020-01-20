import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from datasets.hico_constants import HicoConstants
from utils.bbox_utils import compute_area
from datasets.metadata import coco_classes

def select_det_ids(boxes,scores,nms_keep_ids,score_thresh,max_dets, required=False):
    if nms_keep_ids is None:
        nms_keep_ids = np.arange(0,scores.shape[0])
    
    # Select non max suppressed dets
    nms_scores = scores[nms_keep_ids]
    nms_boxes = boxes[nms_keep_ids]

    # Select dets above a score_thresh and which have area > 1
    nms_ids_above_thresh = np.nonzero(nms_scores > score_thresh)[0]
    nms_ids = []
    for i in range(min(nms_ids_above_thresh.shape[0],max_dets)):
        area = compute_area(nms_boxes[i],invalid=-1)
        if area > 1:
            nms_ids.append(i)
        
    # If no dets satisfy previous criterion select the highest ranking one with area > 1
    if len(nms_ids)==0:
        if required:
            nms_ids.append(np.argmax(nms_scores))
        # for i in range(nms_keep_ids.shape[0]):
        #     area = compute_area(nms_boxes[i],invalid=-1)
        #     if area > 1:
        #         nms_ids = [i]
        #         break
        else:
            return []
    # Convert nms ids to box ids
    nms_ids = np.array(nms_ids,dtype=np.int32)
    try:
        ids = nms_keep_ids[nms_ids]
    except:
        import pdb; pdb.set_trace()

    return ids

def select_dets(
        boxes,
        scores,
        nms_keep_indices,
        exp_const):
    selected_dets = []
    
    start_end_ids = np.zeros([len(coco_classes)-1,2],dtype=np.int32)
    start_id = 0
    for cls_ind, cls_name in enumerate(coco_classes):
        if cls_ind == 0:
            # remove the predictions with background label
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

        # guarantee at least have one person detection
        if cls_name=='person':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.human_score_thresh,
                data_const.max_num_human,
                required=True)
                
        elif cls_name=='__background__':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.background_score_thresh,
                data_const.max_num_background)
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class)
        try:
            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
        except:
            import ipdb; ipdb.set_trace()

        else:
            boxes_scores_rpn_id_label = np.concatenate((
                cls_boxes[select_ids],
                np.expand_dims(cls_scores[select_ids],1),
                np.expand_dims(select_ids,1),
                np.expand_dims([cls_ind] * len(select_ids),1)), 1)

        selected_dets.append(boxes_scores_rpn_id_label)
        num_boxes = boxes_scores_rpn_id_label.shape[0]
        start_end_ids[cls_ind-1,:] = [start_id,start_id+num_boxes]
        start_id += num_boxes

    selected_dets = np.concatenate(selected_dets)
    # guarantee at least have one object  
    object_selected_dets = []
    if selected_dets.shape[0] == 1:
        for cls_ind, cls_name in enumerate(coco_classes):
            if cls_ind == 0 or cls_ind == 1:
                # remove the predictions with background label
                continue
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class,
                required=True)

            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
            else:
                boxes_scores_rpn_id_label = np.concatenate((
                    cls_boxes[select_ids],
                    np.expand_dims(cls_scores[select_ids],1),
                    np.expand_dims(select_ids,1),
                    np.expand_dims([cls_ind] * len(select_ids),1)), 1)

            object_selected_dets.append(boxes_scores_rpn_id_label)

        object_selected_dets = np.concatenate(object_selected_dets)
        max_score_idx = np.argmax(object_selected_dets[:,4])
        object_selected_det = object_selected_dets[max_score_idx,:]
        
        try:
            selected_dets = np.concatenate((selected_dets, object_selected_det[None, :]))
            start_end_ids[int(object_selected_det[6]-1)] = [1,2]
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        
        
    return selected_dets, start_end_ids


def select(data_const):
    io.mkdir_if_not_exists(data_const.proc_dir)
    
    select_boxes_dir = data_const.proc_dir

    # Print where the boxes are coming from and where the output is written
    print(f'Boxes will be read from: {data_const.faster_rcnn_boxes}')
    print(f'Boxes will be written to: {select_boxes_dir}')

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)

    print('Creating selected_coco_cls_dets.hdf5 file ...')
    # hdf5_file = os.path.join(select_boxes_dir,'selected_coco_cls_dets_0.1eval.hdf5')
    hdf5_file = os.path.join(select_boxes_dir,'selected_coco_cls_dets.hdf5')
    f = h5py.File(hdf5_file,'w')

    print('Selecting boxes ...')
    for anno in tqdm(anno_list):
        global_id = anno['global_id']

        # # get more detection for evaluation
        # if 'test' in global_id:
        #     data_const.human_score_thresh = 0.1
        #     data_const.object_score_thresh = 0.1

        boxes_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_boxes.npy')
        boxes = np.load(boxes_npy)
        
        scores_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_scores.npy')
        scores = np.load(scores_npy)
        
        nms_keep_indices_json = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_nms_keep_indices.json')
        nms_keep_indices = io.load_json_object(nms_keep_indices_json)

        # import ipdb; ipdb.set_trace()
        selected_dets, start_end_ids = select_dets(boxes,scores,nms_keep_indices,data_const)
        f.create_group(global_id)
        f[global_id].create_dataset('boxes_scores_rpn_ids',data=selected_dets)
        f[global_id].create_dataset('start_end_ids',data=start_end_ids)
        
    f.close()

if __name__ == "__main__":
    data_const = HicoConstants()
    select(data_const)