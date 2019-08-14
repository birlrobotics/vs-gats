from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import ipdb

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import dgl
import networkx as nx

import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from model.grnn import GRNN
from dataset import metadata


def vis_img(img, bboxs, labels, scores=None, raw_action=None, atten=None):
    try:
        if len(bboxs) == 0:
            return img    

        if scores is not None:
            keep = np.where(scores > 0.7)
            bboxs = bboxs[keep]
            labels = labels[keep]
            scores = scores[keep] 
            if raw_action is not None:
                raw_action = raw_action[keep]

        line_width = 1
        color = (120,0,0)
        # build the Font object
        font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=15)
        for idx, (bbox, label) in enumerate(zip(bboxs, labels)):
            Drawer = ImageDraw.Draw(img)
            # ipdb.set_trace()
            Drawer.rectangle(list(bbox), outline=(225,0,0), width=line_width)
            if raw_action is None:
                text = metadata.coco_classes[label]
                if scores is not None:
                    text = text + " " + '{:.3f}'.format(scores[idx])
                    # text = text + str(idx)
                h, w = font.getsize(text)
                Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
                
            else:         
                action_idx = np.where(raw_action[idx] > 0.1)[0]
                text = str()
                if len(action_idx) > 0:
                    # ipdb.set_trace()
                    for i in range(len(action_idx)):
                        text = text + ' ' +  metadata.action_classes[action_idx[i]] + ' ' + '{:.2f}'.format(raw_action[idx][action_idx[i]]) 
                    h, w = font.getsize(text)
                    Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

def main(args):
    
    # use GPU if available else revert to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)

    # prepare for data
    # ipdb.set_trace()
    if (args.data is not None) and not (args.random_data):
        test_data = pickle.load(open(args.data, 'rb'))
    else:
        data_list = sorted(os.listdir('dataset/processed/train2015'))
        test_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=True)
        # random.shuffle(nums)
        iterator = iter(test_loader)
        test_data = pickle.load(open('dataset/processed/train2015/' + iterator.next()[0], 'rb'))

    img_name = test_data['img_name']
    det_boxes = test_data['boxes']
    roi_labels = test_data['classes']
    roi_scores = test_data['scores']
    node_num = test_data['node_num']
    node_labels = test_data['node_labels']
    features = test_data['feature']
    print("Testing on image named {}".format(img_name))
    if node_num == 0:
        print("No detection. Please test another image!!!")

    # Load checkpoint and set up model
    try:
        # load checkpoint
        checkpoint = torch.load(args.pretrained, map_location=device)
        in_feat, out_feat, hidden_size, action_num  = checkpoint['in_feat'], checkpoint['out_feat'],\
                                                      checkpoint['hidden_size'], checkpoint['action_num']
        print('Checkpoint loaded!')
        # set up model and initialize it with uploaded checkpoint
        model = GRNN(in_feat=in_feat, out_feat=out_feat, hidden_size=hidden_size, action_num=action_num)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)

    # referencing
    features = torch.FloatTensor(features).to(device)
    outputs, atten = model(node_num, features, roi_labels) 
    # show result
    image = Image.open(os.path.join('dataset/hico/images/train2015', img_name)).convert('RGB')
    image_temp = image.copy()
    sigmoid = nn.Sigmoid()
    raw_outputs = sigmoid(outputs)
    raw_outputs = raw_outputs.cpu().detach().numpy()
    # ipdb.set_trace()
    # class_img = vis_img(image, det_boxes, roi_labels, roi_scores)
    class_img = vis_img(image, det_boxes, roi_labels, roi_scores, node_labels)
    action_img = vis_img(image_temp, det_boxes, roi_labels, roi_scores, raw_outputs, atten)
    
    fig = plt.figure(figsize=(100,100))
    fig.suptitle(img_name, fontsize=16)
    ax1 = plt.subplot(1,2,1)
    ax1.set_title('class_detection')
    plt.imshow(class_img)
    plt.axis('off')
    ax2 = plt.subplot(1,2,2)
    ax2.set_title('action_detection')
    plt.imshow(action_img)
    plt.axis('off')
    plt.show()
    # ipdb.set_trace()

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
    parser = argparse.ArgumentParser(description='inference of the model')

    parser.add_argument('--data', type=str, default='./dataset/processed/train2015/HICO_train2015_00010000.p',
                        help='A path to the test data is necessary.')
    parser.add_argument('--dataset', '-d', type=str, default='ucf101', choices=['ucf101','hmdb51'],
                        help='Location of the dataset: ucf101')
    parser.add_argument('--pretrained', '-p', type=str, default='./checkpoints/v1/checkpoint_200_epoch.pth',
                        help='Location of the checkpoint file: ./checkpoints/v1/checkpoint_150_epoch.pth')
    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')
    parser.add_argument('--random_data', type=str2bool, default='false',
                        help='select data randomly from the test dataset: true')
    args = parser.parse_args()
    # inferencing
    main(args)