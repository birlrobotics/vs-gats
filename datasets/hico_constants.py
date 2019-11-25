import os

import utils.io as io

class HicoConstants(io.JsonSerializableClass):
    def __init__(
            self,
            clean_dir=os.path.join(os.getcwd(),'datasets/hico'),
            proc_dir=os.path.join(os.getcwd(),'datasets/processed/hico'),
            res_dir=os.path.join(os.getcwd(),'result/hico'),
            feat_type='fc7',
            exp_ver ='test'):
        self.clean_dir = clean_dir
        self.proc_dir = proc_dir
        self.hico = res_dir
        self.result_dir = res_dir + '/' + exp_ver
        self.feat_type = feat_type

        # Clean constants
        self.anno_bbox_mat = os.path.join(self.clean_dir,'anno_bbox.mat')
        self.anno_mat = os.path.join(self.clean_dir,'anno.mat')
        self.hico_list_hoi_txt = os.path.join(
            self.clean_dir,
            'hico_list_hoi.txt')
        self.hico_list_obj_txt = os.path.join(
            self.clean_dir,
            'hico_list_obj.txt')
        self.hico_list_vb_txt = os.path.join(
            self.clean_dir,
            'hico_list_vb.txt')
        self.images_dir = os.path.join(self.clean_dir,'images')

        # Processed constants
        self.anno_list_json = os.path.join(self.proc_dir,'anno_list.json')
        self.hoi_list_json = os.path.join(self.proc_dir,'hoi_list.json')
        self.object_list_json = os.path.join(self.proc_dir,'object_list.json')
        self.verb_list_json = os.path.join(self.proc_dir,'verb_list.json')

        # Need to run split_ids.py
        self.split_ids_json = os.path.join(self.proc_dir,'split_ids.json')

        # Need to run hoi_cls_count.py
        self.hoi_cls_count_json = os.path.join(self.proc_dir,'hoi_cls_count.json')
        self.bin_to_hoi_ids_json = os.path.join(self.proc_dir,'bin_to_hoi_ids.json')
        # path to keep the detection from faster-rcnn
        self.faster_rcnn_boxes = os.path.join(self.proc_dir,'faster_rcnn_boxes')
        self.faster_det_fc7_feat = os.path.join(self.faster_rcnn_boxes, 'faster_rcnn_fc7.hdf5')
        self.faster_det_pool_feat = os.path.join(self.faster_rcnn_boxes, 'faster_rcnn_pool.hdf5')

        # select proper boxes from rpn
        self.background_score_thresh = 0.4
        self.human_score_thresh = 0.8
        self.object_score_thresh = 0.3
        self.max_num_background = 10
        self.max_num_human = 10
        self.max_num_objects_per_class = 10
        self.boxes_scores_rpn_ids_labels = os.path.join(self.proc_dir, 'selected_coco_cls_dets.hdf5')

        # set the iou thresh to evaluate instance detection
        self.iou_thresh = 0.5

        # train_val_test data
        self.bad_faster_rcnn_det_ids = os.path.join('result', 'bad_faster_rcnn_det_imgs.json')
        
        if self.feat_type == 'fc7':
            self.hico_trainval_data = os.path.join(self.proc_dir, 'hico_trainval_data_fc7_edge.hdf5')
            self.hico_test_data = os.path.join(self.proc_dir, 'hico_test_data_fc7_edge.hdf5')
        else:
            self.hico_trainval_data = os.path.join(self.proc_dir, 'hico_trainval_data_pool.hdf5')
            self.hico_test_data = os.path.join(self.proc_dir, 'hico_test_data_pool.hdf5')
        
        # spatial features
        self.trainval_spatial_feat = os.path.join(self.proc_dir, 'trainval_spatial_features.hdf5')
        self.test_spatial_feat = os.path.join(self.proc_dir, 'test_spatial_features.hdf5')

        # word2vec
        self.word2vec = os.path.join(self.proc_dir, 'hico_word2vec.hdf5')

        # inference directory
        self.infer_dir = './inference_imgs'