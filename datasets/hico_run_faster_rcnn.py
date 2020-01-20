import os
import ipdb
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import utils.io as io
from datasets.hico_constants import HicoConstants
import h5py
import json

import torchvision
import torch
# from utils.vis_tool import vis_img

if __name__ == "__main__":
    # set up model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, \
                                                                 box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3)
    device = torch.device('cuda:0')
    model.cuda()
    model.eval()

    print('Begining...')
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    io.mkdir_if_not_exists(data_const.faster_rcnn_boxes, recursive=True)

    fc7_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_fc7.hdf5')
    pool_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_pool.hdf5')
    fc7_feat = h5py.File(fc7_feat_hdf5, 'w')
    pool_feat = h5py.File(pool_feat_hdf5, 'w')

    for ind in tqdm(range(len(anno_list))):
        save_dir = data_const.faster_rcnn_boxes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        root = 'datasets/hico/images/'
        image = Image.open(os.path.join(root, anno_list[ind]['image_path_postfix'])).convert('RGB')
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(device)
        outputs = model([input], save_feat=True)

        # save object detection result data
        np.save(os.path.join(save_dir, '{}_boxes.npy'.format(anno_list[ind]['global_id'])), outputs[0]['boxes'].cpu().detach().numpy())
        np.save(os.path.join(save_dir, '{}_scores.npy'.format(anno_list[ind]['global_id'])), outputs[0]['scores'].cpu().detach().numpy())
        nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(anno_list[ind]['global_id']))
        io.dump_json_object(outputs[0]['labels'], nms_keep_indices_path)
        fc7_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['fc7_feat'].cpu().detach().numpy())
        pool_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['pool_feat'].cpu().detach().numpy())

    fc7_feat.close()
    pool_feat.close()
    print('Make detection data successfully!')