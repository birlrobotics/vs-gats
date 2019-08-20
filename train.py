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
import h5py
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from model.model import AGRNN
from datasets import metadata
import utils.io as io
from utils.vis_tool import vis_img
from datasets.hico_constants import HicoConstants

###########################################################################################
#                                     TRAIN/TEST MODEL                                    #
###########################################################################################

def run_model(args, data_const):
    # get data file
    trainval_list = io.load_json_object(data_const.split_ids_json) 
    dataset = {'train': trainval_list['train'], 'val': trainval_list['val']}
    print('load split_ids list successfully')
    # use default DataLoader() to load the data. In this case, the dataset is a list 
    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=1, shuffle=True)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=1, shuffle=True)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    print('set up dataloader successfully')

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('training on {}...'.format(device))

    model = AGRNN()
    model.to(device)

    # # build optimizer && criterion  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1) #the scheduler divides the lr by 10 every 150 epochs

    if args.train_model == 'epoch':
        epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const)
    else:
        iteration_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const)

def epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const):
    print('epoch training...')
    
    trainval_data = h5py.File(data_const.hico_trainval_data, 'r')
    print('load train data successfully...')
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'epoch_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'epoch_train'), recursive=True)

    for epoch in range(args.epoch):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = time.time()
            running_loss = 0.0
            idx = 0
            # for idx in tqdm(range((dataset[phase]))):
            # for idx, file in tqdm(enumerate(dataloader[phase])): 
            for file in tqdm(dataloader[phase]): 
                if file[0] not in trainval_data.keys(): continue
                train_data = trainval_data[file[0]]
                img_name = file[0] + '.jpg'
                det_boxes = train_data['boxes'][:]
                roi_labels = train_data['classes'][:]
                roi_scores = train_data['scores'][:]
                node_num = train_data['node_num'].value
                node_labels = train_data['node_labels'][:]
                features = train_data['feature'][:]
                if node_num == 0 or node_num == 1: continue
                # ipdb.set_trace()    
                features, node_labels = torch.FloatTensor(features).to(device), torch.FloatTensor(node_labels).to(device)

                # nx.draw(graph.to_networkx(), node_size=500, with_labels=True, node_color='#00FFFF')
                # plt.show()
                # ipdb.set_trace()
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    outputs, atten = model(node_num, features, roi_labels, feat_type='fc7')
                    loss = criterion(outputs, node_labels)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    # turn off the gradients for vaildation, save memory and computations
                    with torch.no_grad():
                        outputs, atten = model(node_num, features, roi_labels)
                        loss = criterion(outputs, node_labels)

                    # print result every 1000 iterationa during validation
                    if idx==0 or idx%1000==999:
                        image = Image.open(os.path.join(args.img_data, img_name)).convert('RGB')
                        image_temp = image.copy()
                        sigmoid = nn.Sigmoid()
                        raw_outputs = sigmoid(outputs)
                        raw_outputs = raw_outputs.cpu().detach().numpy()
                        # class_img = vis_img(image, det_boxes, roi_labels, roi_scores)
                        class_img = vis_img(image, det_boxes, roi_labels, roi_scores, node_labels.cpu().numpy())
                        action_img = vis_img(image_temp, det_boxes, roi_labels, roi_scores, raw_outputs)
                        writer.add_image('gt_detection', np.array(class_img).transpose(2,0,1))
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
                          'lr': args.lr,
                  'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

    writer.close()
    print('Finishing training!')

def iteration_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const):
    print('iteration training...')
    trainval_data = h5py.File(data_const.hico_trainval_data, 'r')
    print('load train data successfully...')

    # # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'iteration_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'iteration_train'), recursive=True)
    iter=0
    for epoch in range(args.epoch):
        start_time = time.time()
        running_loss = 0.0
        for file in tqdm(dataloader['train']): 
            if file[0] not in trainval_data.keys(): continue
            train_data = train_data = trainval_data[file[0]]
            img_name = file[0] + '.jpg'
            det_boxes = train_data['boxes'][:]
            roi_labels = train_data['classes'][:]
            roi_scores = train_data['scores'][:]
            node_num = train_data['node_num'].value
            node_labels = train_data['node_labels'][:]
            features = train_data['feature'][:]
            if node_num == 0 or node_num == 1: continue
            features, node_labels = torch.FloatTensor(features).to(device), torch.FloatTensor(node_labels).to(device)
            # training
            model.train()
            model.zero_grad()
            outputs, atten = model(node_num, features, roi_labels)
            loss = criterion(outputs, node_labels)
            loss.backward()
            optimizer.step()
            # loss.backward()
            # if step%exp_const.imgs_per_batch==0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            # accumulate loss of each batch
            running_loss += loss.item() * node_labels.shape[0]
            if iter % 99 == 0:
                loss = running_loss/(iter+1)
                writer.add_scalar('train_loss_iter', loss, iter)

            if iter % 4999 == 0:
                num_samples = 2500
                val_loss = 0
                idx = 0
                for file in tqdm(dataloader['val']):
                    if file not in trainval_data.keys(): continue
                    # if idx > num_samples:
                    #     break
                    train_data = train_data = trainval_data[file]
                    img_name = file[0] + '.jpg'
                    det_boxes = train_data['boxes'][:]
                    roi_labels = train_data['classes'][:]
                    roi_scores = train_data['scores'][:]
                    node_num = train_data['node_num'].value
                    node_labels = train_data['node_labels'][:]
                    features = train_data['feature'][:]
                    if node_num == 1: 
                        continue 

                    features, node_labels = torch.FloatTensor(features).to(device), torch.FloatTensor(node_labels).to(device)
                    # training
                    model.eval()
                    model.zero_grad()
                    outputs, atten = model(node_num, features, roi_labels)
                    loss = criterion(outputs, node_labels)
                    val_loss += loss.item() * node_labels.shape[0]

                    if idx==0 or idx%1000 == 999:
                        image = Image.open(os.path.join(args.img_data, img_name)).convert('RGB')
                        image_temp = image.copy()
                        sigmoid = nn.Sigmoid()
                        raw_outputs = sigmoid(outputs)
                        raw_outputs = raw_outputs.cpu().detach().numpy()
                        # class_img = vis_img(image, det_boxes, roi_labels, roi_scores)
                        class_img = vis_img(image, det_boxes, roi_labels, roi_scores, node_labels.cpu().numpy())
                        action_img = vis_img(image_temp, det_boxes, roi_labels, roi_scores, raw_outputs)
                        writer.add_image('gt_detection', np.array(class_img).transpose(2,0,1))
                        writer.add_image('action_detection', np.array(action_img).transpose(2,0,1))
                    idx+=1
                loss = val_loss / len(dataset['val'])
                writer.add_scalar('val_loss_iter', loss, iter)
                
                # save model
                checkpoint = { 
                            'lr': args.lr,
                        'in_feat': 2*1024, 
                        'out_feat': 1024,
                    'hidden_size': 1024,
                    'action_num': 117,
                    'state_dict': model.state_dict()
                }
                save_name = "checkpoint_" + str(iter+1) + '_iters.pth'
                torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'iteration_train', save_name))

            iter+=1

        epoch_loss = running_loss / len(dataset['train'])
        if (epoch % args.print_every) == 0:
            end_time = time.time()
            # print("[{}] Epoch: {}/{} Loss: {} Acc: {} Execution time: {}".format(\
            #         phase, epoch+1, args.epoch, epoch_loss, epoch_acc, (end_time-start_time)))
            print("[{}] Epoch: {}/{} Loss: {} Execution time: {}".format(\
                    'train', epoch+1, args.epoch, epoch_loss, (end_time-start_time)))

    writer.close()
    print('Finishing training!')



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

parser.add_argument('--img_data', type=str, default='datasets/hico/images/train2015',
                    help='location of the original dataset')
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

parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', required=True,
                    help='the version of code, will create subdir in log/ && checkpoints/ ')

parser.add_argument('--train_model', '--t_m', type=str, default='epoch', required=True,
                    choices=['epoch', 'iteration'],
                    help='the version of code, will create subdir in log/ && checkpoints/ ')

args = parser.parse_args() 

if __name__ == "__main__":
    data_const = HicoConstants()
    run_model(args, data_const)


