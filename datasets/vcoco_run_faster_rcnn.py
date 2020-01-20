# import os
# dir_path = os.path.dirname(__file__)
# import sys
# sys.path.append(os.path.join(dir_path, 'vcoco'))
# import sys
# sys.path.append('./vcoco')
import os
import h5py
import ipdb
from tqdm import tqdm
from PIL import Image

from datasets.vcoco import vsrl_utils as vu
import utils.io as io
from datasets.vcoco_constants import VcocoConstants

import torchvision
import torch

if __name__ == '__main__':
    # set up the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, \
                                                                 box_score_thresh=0.1, box_nms_thresh=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    data_const = VcocoConstants()
    # Load COCO annotations for V-COCO images
    coco = vu.load_coco()
    for subset in ["vcoco_train", "vcoco_test", "vcoco_val"]:
        # create the folder/file to save corresponding detection results
        io.mkdir_if_not_exists(os.path.join(data_const.proc_dir, subset), recursive=True)
        faster_rcnn_det_hdf5 = os.path.join(data_const.proc_dir, subset, 'faster_rcnn_det.hdf5')
        faster_rcnn_det_data = h5py.File(faster_rcnn_det_hdf5, 'w')       
    
        # load the VCOCO annotations for image set
        print('Construct object detection results for {} dataset'.format(subset.split('_')[1]))
        vcoco = vu.load_vcoco(subset)
        img_id_list = vcoco[0]['image_id'][:,0].tolist()
        nms_keep_indices_dict = {}
        # ipdb.set_trace()
        for img_id in tqdm(set(img_id_list)):
            img_path = os.path.join('datasets/vcoco/coco/images', coco.loadImgs(ids=img_id)[0]['coco_url'].split('.org')[1][1:])
            img = Image.open(img_path).convert('RGB')
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            img_tensor = img_tensor.to(device)
            outputs = model([img_tensor], save_feat=True)
            # save object detection results
            faster_rcnn_det_data.create_group(str(img_id))
            faster_rcnn_det_data[str(img_id)].create_dataset(name='boxes', data=outputs[0]['boxes'].cpu().detach().numpy()) 
            faster_rcnn_det_data[str(img_id)].create_dataset(name='scores', data=outputs[0]['scores'].cpu().detach().numpy())  
            faster_rcnn_det_data[str(img_id)].create_dataset(name='fc7_feaet', data=outputs[0]['fc7_feat'].cpu().detach().numpy()) 
            faster_rcnn_det_data[str(img_id)].create_dataset(name='pool_feaet', data=outputs[0]['pool_feat'].cpu().detach().numpy())
            nms_keep_indices_dict[str(img_id)] = outputs[0]['labels']
        faster_rcnn_det_data.close()
        io.dump_json_object(nms_keep_indices_dict, os.path.join(data_const.proc_dir, subset, 'nms_keep_indices.json'))
    print('Finished!!!')
