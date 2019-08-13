from __future__ import print_function
import numpy as np
import argparse
import os
import glob
import time
from tqdm import tqdm
import ipdb

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader
import dgl
import networkx as nx

import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from model.grnn import GRNN
from dataset import metadata

###########################################################################################
#                                 SET SOME ARGUMENTS                                      #
###########################################################################################
# define a string2boolean type function for argparse
def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

parser = argparse.ArgumentParser(description="separable 3D CNN for action classification!")

parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size: 10')
parser.add_argument('--clip_len', type=int, default=64,
                    help='set time step: 64') 
parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='dropout parameter: 0.2')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate: 0.001')
parser.add_argument('--gpu', type=str2bool, default='true', 
                    help='chose to use gpu or not: True') 
parser.add_argument('--test', type=str2bool, default='true',
                    help="test the model during traing: True")
#  parser.add_argument('--clip', type=int, default=4,
#                      help='gradient clipping: 4')

parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101','hmdb51'],
                    help='location of the dataset: ucf101')
parser.add_argument('--pretrained', type=str, default='',
                    help='location of the pretrained model file for training: None')
parser.add_argument('--log_dir', type=str, default='./log',
                    help='path to save the log data like loss\accuracy... : ./log') 
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='path to save the checkpoints: ./checkpoints')

parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train: 300') 
parser.add_argument('--print_every', type=int, default=10,
                    help='number of steps for printing training and validation loss: 10') 
parser.add_argument('--save_every', type=int, default=50,
                    help='number of steps for saving the model parameters: 50')                      
parser.add_argument('--test_every', type=int, default=50,
                    help='number of steps for testing the model: 50') 
# for dataset processing
parser.add_argument('--resize_height',  type=int, default=256,
                    help='resize the height of frames before processing: 256')
parser.add_argument('--resize_width',  type=int, default=256,
                    help='resize the width of frames before processing: 256') 
parser.add_argument('--crop_height',  type=int, default=224,
                    help='crop the height of frames when processing: 224')
parser.add_argument('--crop_width',  type=int, default=224,
                    help='crop the widht of frames when processing: 224')   

args = parser.parse_args() 

###########################################################################################

def vis_img(img, bboxs, labels, scores=None, raw_action=None):
    try:
        if len(bboxs) == 0:
            return img    

        if scores is not None:
            keep = np.where(scores > 0.8)
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
            Drawer.rectangle(list(bbox), outline=(120,0,0), width=line_width)
            if raw_action is None:
                text = metadata.hico_classes[label]
                if scores is not None:
                    text = text + " " + '{:.3f}'.format(scores[idx])
                h, w = font.getsize(text)
                Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
                
            else:
                action_idx = np.where(raw_action[idx] > 0.5)[0]
                text = str()
                if len(action_idx) > 0:
                    for i in range(len(action_idx)):
                        text = text + " " + metadata.action_classes[action_idx[i]]
                        h, w = font.getsize(text)
                        Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                    Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

###########################################################################################
#                                     TRAIN/TEST MODEL                                    #
###########################################################################################
def run_model(args):

    img_dir = 'dataset/hico/images/train2015'
    # get data file
    data_list = sorted(os.listdir('dataset/processed/train2015/'))
    train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)
    dataset = {'train': train_list, 'val': val_list}
    # use default DataLoader() to load the data. In this case, the dataset is a list 
    train_dataloader = DataLoader(dataset=train_list, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(dataset=val_list, batch_size=1, shuffle=True)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('training on {}'.format(device))

    model = GRNN(in_feat=2*1024, out_feat=1024, hidden_size=1024, action_num=117)
    model.to(device)

    # # build optimizer && criterion  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1) #the scheduler divides the lr by 10 every 150 epochs

    # # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for epoch in range(args.epoch):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = time.time()
            running_loss = 0.0
            idx = 0
            # for idx in tqdm(range((dataset[phase]))):
            # for idx, file in tqdm(enumerate(dataloader[phase])): 
            for file in tqdm(dataloader[phase]): 
                train_data = pickle.load(open('dataset/processed/train2015/{}'.format(file[0]), 'rb'))
                img_name = train_data['img_name']
                det_boxes = train_data['boxes']
                roi_labels = train_data['classes']
                roi_scores = train_data['scores']
                node_num = train_data['node_num']
                node_labels = train_data['node_labels']
                features = train_data['feature']
                if node_num == 0: 
                    continue
                # ipdb.set_trace() 
                features, node_labels = torch.FloatTensor(features).to(device), torch.FloatTensor(node_labels).to(device)

                # nx.draw(graph.to_networkx(), node_size=500, with_labels=True, node_color='#00FFFF')
                # plt.show()
                # ipdb.set_trace()
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    outputs, atten = model(node_num, features, roi_labels)
                    loss = criterion(outputs, node_labels)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    # turn off the gradients for vaildation, save memory and computations
                    with torch.no_grad():
                        outputs, atten = model(node_num, features, roi_labels)
                        loss = criterion(outputs, node_labels)

                    # print resulr every 1000 iterationa during validation
                    if idx==0 or idx%1000 == 999:
                        image = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
                        image_temp = image.copy()
                        sigmoid = nn.Sigmoid()
                        raw_outputs = sigmoid(outputs)
                        raw_outputs = raw_outputs.cpu().detach().numpy()
                        class_img = vis_img(image, det_boxes, roi_labels, roi_scores)
                        action_img = vis_img(image_temp, det_boxes, roi_labels, roi_scores, raw_outputs)
                        writer.add_image('class_detection', np.array(class_img).transpose(2,0,1))
                        writer.add_image('action_detection', np.array(action_img).transpose(2,0,1))

                idx+=1
                # accumulate loss of each batch
                running_loss += loss.item() * node_labels.shape[0]
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataset[phase])
            
            # log trainval datas, and visualize them in the same graph
            if phase == 'train':
                train_loss = epoch_loss  
        
            else:
                writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
                
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                # print("[{}] Epoch: {}/{} Loss: {} Acc: {} Execution time: {}".format(\
                #         phase, epoch+1, args.epoch, epoch_loss, epoch_acc, (end_time-start_time)))
                print("[{}] Epoch: {}/{} Loss: {} Execution time: {}".format(\
                        phase, epoch+1, args.epoch, epoch_loss, (end_time-start_time)))

        # scheduler.step()
        # save model
        if epoch % args.save_every == (args.save_every -1):
            checkpoint = { 
                     'in_feat': 2*1024, 
                    'out_feat': 1024,
                 'hidden_size': 1024,
                  'action_num': 117,
                  'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, save_name))

    writer.close()
    print('Finishing training!')
  
if __name__ == "__main__":
    run_model(args)


