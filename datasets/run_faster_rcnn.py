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

def vis_img(img, bboxs, labels, scores=None):
    try:
        if len(bboxs)== 0:
            return img    

        if scores is not None:
            keep = np.where(scores > 0.1)
            bboxs = bboxs[keep]
            labels = labels[keep]
            scores = scores[keep]
        
        score_idx = 0 
        line_width = 1
        for (bbox, label) in zip(bboxs, labels):
            Drawer = ImageDraw.Draw(img)
            # ipdb.set_trace()
            text = str()
            Drawer.rectangle(list(bbox), outline='red', width=line_width)
            text = COCO_INSTANCE_CATEGORY_NAMES[label]
            if scores is None:
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red')
            else:
                text = text + " " + '{:.3f}'.format(scores[score_idx])
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red')
                score_idx +=1
        return img

    except Exception as e:
        print("Error:" ,e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

if __name__ == "__main__":
    # set up model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3)
    devise = torch.device('cuda:0')
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
        # image = Image.open(os.path.join(root, 'train2015','HICO_train2015_00001879.jpg')).convert('RGB')
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(devise)
        outputs = model([input], save_feat=True)
        # img = vis_img(image, outputs[1][0]['boxes'].cpu().detach().numpy(), outputs[1][0]['labels'].cpu().detach().numpy(), outputs[1][0]['scores'].cpu().detach().numpy())
        # plt.subplot(1,1,1)
        # plt.imshow(np.array(img))
        # plt.axis('off')
        # plt.ion()
        # plt.pause(10)
        # plt.close()
        # ipdb.set_trace()
        # np.save(os.path.join(save_dir, '{}_fc7.npy'.format(anno_list[ind]['global_id'])), outputs[0]['box_feat'].cpu().detach().numpy())
        np.save(os.path.join(save_dir, '{}_boxes.npy'.format(anno_list[ind]['global_id'])), outputs[0]['boxes'].cpu().detach().numpy())
        np.save(os.path.join(save_dir, '{}_scores.npy'.format(anno_list[ind]['global_id'])), outputs[0]['scores'].cpu().detach().numpy())
        nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(anno_list[ind]['global_id']))
        # with open(nms_keep_indices_path,'w') as file:
        #     json.dump(outputs[0]['labels'], file)
        io.dump_json_object(outputs[0]['labels'], nms_keep_indices_path)
        fc7_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['fc7_feat'].cpu().detach().numpy())
        pool_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['pool_feat'].cpu().detach().numpy())

    fc7_feat.close()
    pool_feat.close()
    print('Make detection data successfully!')

    # print('Begining...')
    # for data in ['train2015', 'test2015']:
    #     # if data == 'train2015': continue
    #     save_dir = os.path.join('detection_data', data) 
    #     # save_dir = '/home/birl/ml_dl_projects/bigjun/hio/no_frills_hoi_det/data_symlinks/hico_processed/faster_rcnn_boxes'
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     root = 'hico/images/' + data
    #     file_list = sorted(os.listdir(root))
    #     for idx in tqdm(range(len(file_list))):
    #         image = Image.open(os.path.join(root, file_list[idx])).convert('RGB')
    #         # plt.figure(figsize=(100,100))
    #         # plt.subplot(1,2,1)
    #         # plt.imshow(np.array(image))
    #         # plt.axis('off')
    #         # plt.xticks([])
    #         # plt.yticks([])
    #         # ipdb.set_trace()
    #         input = torchvision.transforms.functional.to_tensor(image)
    #         input = input.to(devise)
    #         outputs = model([input], save_feat=True)
    #         # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['pool'], output_size=7, sampling_ratio=2)   #featmap_names: 0 1 2 3 'pool'
    #         # roi_feat = roi_pooler(outputs[2], [outputs[0]], outputs[3])
    #         roi_feat = model.roi_heads.box_roi_pool(outputs[2], [outputs[0]], outputs[3])
    #         save_feat = model.roi_heads.box_head(roi_feat)
    #         np.save(os.path.join(save_dir, '{}_boxes_feature.npy'.format(file_list[idx].split('.')[0])), save_feat.cpu().detach().numpy())
    #         np.save(os.path.join(save_dir, '{}_det_class.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['labels'].cpu().detach().numpy())
    #         np.save(os.path.join(save_dir, '{}_det_boxes.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['boxes'].cpu().detach().numpy())
    #         np.save(os.path.join(save_dir, '{}_det_scores.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['scores'].cpu().detach().numpy())
    #         # np.save(os.path.join(save_dir, '{}_fc7.npy'.format(file_list[idx].split('.')[0])), outputs[0]['box_feat'].cpu().detach().numpy())
    #         # np.save(os.path.join(save_dir, '{}_boxes.npy'.format(file_list[idx].split('.')[0])), outputs[0]['boxes'].cpu().detach().numpy())
    #         # np.save(os.path.join(save_dir, '{}_scores.npy'.format(file_list[idx].split('.')[0])), outputs[0]['scores'].cpu().detach().numpy())
    #         # nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(file_list[idx].split('.')[0]))
    #         # with open(nms_keep_indices_path,'w') as file:
    #         #     json.dump(outputs[0]['labels'], file)
    # print('Make detection data successfully!')
    #         # ipdb.set_trace()
    #         # print(len(outputs[1][0]["labels"]))
    #         # img = vis_img(image, outputs[1][0]['boxes'].cpu().detach().numpy(), outputs[1][0]['labels'].cpu().detach().numpy(), outputs[1][0]['scores'].cpu().detach().numpy())
    #         # plt.subplot(1,1,1)
    #         # plt.imshow(np.array(img))
    #         # # plt.xticks([])
    #         # # plt.yticks([])
    #         # plt.axis('off')
    #         # plt.ion()
    #         # plt.pause(5)
    #         # plt.close()

