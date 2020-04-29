import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from datasets.vcoco_constants import VcocoConstants
from utils.bbox_utils import compute_area
from datasets.vcoco_metadata import coco_classes
from datasets.vcoco import vsrl_utils as vu

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
                
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class)

        if len(select_ids)==0 :
            boxes_scores_rpn_id_label = np.empty((0,7))

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

    for subset in ["vcoco_train", "vcoco_test", "vcoco_val"]:
        # create the folder/file to save corresponding detection results
        print('Select detection results for {} dataset'.format(subset.split('_')[1]))
        subset_dir = os.path.join(data_const.proc_dir, subset)
        io.mkdir_if_not_exists(subset_dir, recursive=True)

        print(f'Creating selected_coco_cls_dets.hdf5 file for {subset}...')
        hdf5_file = os.path.join(subset_dir,'selected_coco_cls_dets.hdf5')
        f = h5py.File(hdf5_file,'w')

        # Load the VCOCO annotations for image set
        vcoco = vu.load_vcoco(subset)
        img_id_list = vcoco[0]['image_id'][:,0].tolist()

        # Load faster-rcnn detection results
        all_faster_rcnn_det_data = h5py.File(os.path.join(subset_dir, 'faster_rcnn_det.hdf5'), 'r')
        all_nms_keep_indices = io.load_json_object(os.path.join(subset_dir, 'nms_keep_indices.json'))
        print('Selecting boxes ...')
        for img_id in tqdm(set(img_id_list)):

            boxes = all_faster_rcnn_det_data[str(img_id)]['boxes']
            scores = all_faster_rcnn_det_data[str(img_id)]['scores']
            features = all_faster_rcnn_det_data[str(img_id)]['fc7_feaet']
            nms_keep_indices = all_nms_keep_indices[str(img_id)]

            # import ipdb; ipdb.set_trace()
            selected_dets, start_end_ids = select_dets(boxes,scores,nms_keep_indices,data_const)

            selected_feat = []
            for rpn_id in selected_dets[:, 5]:
                selected_feat.append(np.expand_dims(features[rpn_id, :], 0))
            selected_feat = np.concatenate(selected_feat, axis=0
            )
            f.create_group(str(img_id))
            f[str(img_id)].create_dataset('boxes_scores_rpn_ids',data=selected_dets)
            f[str(img_id)].create_dataset('start_end_ids',data=start_end_ids)
            f[str(img_id)].create_dataset('features',data=selected_feat)
            
        f.close()

if __name__ == "__main__":
    data_const = VcocoConstants()
    select(data_const)