import os

import utils.io as io

class VcocoConstants(io.JsonSerializableClass):
    def __init__(
            self,
            proc_dir=os.path.join(os.getcwd(),'datasets/processed/vcoco'),
            res_dir=os.path.join(os.getcwd(),'result/vcoco'),
            feat_type='fc7',
            exp_ver ='test'):
        self.proc_dir = proc_dir
        self.hico = res_dir
        self.result_dir = res_dir + '/' + exp_ver
        self.feat_type = feat_type

        # select proper boxes from rpn
        self.background_score_thresh = 0.4
        self.human_score_thresh = 0.8
        self.object_score_thresh = 0.4  # 0.3
        self.max_num_background = 10
        self.max_num_human = 10
        self.max_num_objects_per_class = 10
        self.boxes_scores_rpn_ids_labels = os.path.join(self.proc_dir, 'selected_coco_cls_dets.hdf5')

        # set the iou thresh to evaluate instance detection
        self.iou_thresh = 0.5

        # original train_val_test image data
        self.original_image_dir = 'datasets/vcoco/coco/images'

        self.original_data_dir = 'datasets/vcoco'
        
        # word2vec
        self.word2vec = os.path.join(self.proc_dir, 'vcoco_word2vec.hdf5')

        # visual && spatial data
        self.train_visual_data = os.path.join(self.proc_dir, 'vcoco_train', 'vcoco_data.hdf5')
        self.train_spatial_data = os.path.join(self.proc_dir, 'vcoco_train', 'spatial_feat.hdf5')
        self.val_visual_data = os.path.join(self.proc_dir, 'vcoco_val', 'vcoco_data.hdf5')
        self.val_spatial_data = os.path.join(self.proc_dir, 'vcoco_val', 'spatial_feat.hdf5')

        # inference directory
        self.infer_dir = './inference_imgs'
