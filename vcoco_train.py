from __future__ import print_function

import os
import time

import dgl
import networkx as nx
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import ipdb
import h5py
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random

import utils.io as io
from model.vcoco_model import AGRNN, Predictor
from datasets import vcoco_metadata
from utils.vis_tool import vis_img_vcoco
from datasets.vcoco_constants import VcocoConstants
from datasets.vcoco_dataset import VcocoDataset, collate_fn

###########################################################################################
#                                     TRAIN/TEST MODEL                                    #
###########################################################################################

def run_model(args, data_const):
    # set up dataset variable
    train_dataset = VcocoDataset(data_const=data_const, subset='vcoco_train', data_aug=args.data_aug, sampler=args.sampler)
    val_dataset = VcocoDataset(data_const=data_const, subset='vcoco_val', data_aug=False, sampler=args.sampler)
    dataset = {'train': train_dataset, 'val': val_dataset}
    print('set up dataset variable successfully')
    # use default DataLoader() to load the data. 
    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    print('set up dataloader successfully')

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('training on {}...'.format(device))

    model = AGRNN(feat_type=args.feat_type, bias=args.bias, bn=args.bn, dropout=args.drop_prob, multi_attn=args.multi_attn, layer=args.layers, diff_edge=args.diff_edge, HICO=args.hico)

    # load pretrained model of HICO_DET dataset
    if args.hico:
        print(f"loading pretrained model of HICO_DET dataset {args.hico}")
        checkpoints = torch.load(args.hico, map_location=device)
        # import ipdb; ipdb.set_trace()
        model.load_state_dict(checkpoints['state_dict'])
        # change the last layer 117->24
        model.edge_readout.classifier.layers[1] = Predictor(model.CONFIG1).classifier.layers[1]
     
    # calculate the amount of all the learned parameters
    parameter_num = 0
    for param in model.parameters():
        parameter_num += param.numel()
    print(f'The parameters number of the model is {parameter_num / 1e6} million')

    # load pretrained model
    if args.pretrained:
        print(f"loading pretrained model {args.pretrained}")
        checkpoints = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
    model.to(device)
    # # build optimizer && criterion  
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    # ipdb.set_trace()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.3) #the scheduler divides the lr by 10 every 150 epochs

    # get the configuration of the model and save some key configurations
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver), recursive=True)
    for i in range(args.layers):
        if i==0:
            model_config = model.CONFIG1.save_config()
            model_config['lr'] = args.lr
            model_config['bs'] = args.batch_size
            model_config['layers'] = args.layers
            model_config['multi_attn'] = args.multi_attn
            model_config['data_aug'] = args.data_aug
            model_config['drop_out'] = args.drop_prob
            model_config['optimizer'] = args.optim
            model_config['diff_edge'] = args.diff_edge
            model_config['model_parameters'] = parameter_num
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l1_config.json'))
        elif i==1:
            model_config = model.CONFIG2.save_config()
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l2_config.json'))
        else:
            model_config = model.CONFIG3.save_config()
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l3_config.json'))
    print('save key configurations successfully...')

    if args.train_model == 'epoch':
        epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const)
    else:
        iteration_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const)

def epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const):
    print('epoch training...')
    
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'epoch_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'epoch_train'), recursive=True)

    for epoch in range(args.start_epoch, args.epoch):
        # each epoch has a training and validation step
        epoch_loss = 0
        for phase in ['train', 'val']:
            start_time = time.time()
            running_loss = 0
            # all_edge = 0
            idx = 0
            
            VcocoDataset.data_sample_count=0
            for data in tqdm(dataloader[phase]): 
                train_data = data
                img_name = train_data['img_name']
                det_boxes = train_data['det_boxes']
                roi_labels = train_data['roi_labels']
                roi_scores = train_data['roi_scores']
                node_num = train_data['node_num']
                edge_labels = train_data['edge_labels']
                edge_num = train_data['edge_num']
                features = train_data['features']
                spatial_feat = train_data['spatial_feat']
                word2vec = train_data['word2vec']
                features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)
                if idx == 10: break    
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    outputs = model(node_num, features, spatial_feat, word2vec, roi_labels)
                    # import ipdb; ipdb.set_trace()
                    loss = criterion(outputs, edge_labels.float())
                    loss.backward()
                    optimizer.step()

                else:
                    model.eval()
                    # turn off the gradients for validation, save memory and computations
                    with torch.no_grad():
                        outputs = model(node_num, features, spatial_feat, word2vec, roi_labels, validation=True)
                        loss = criterion(outputs, edge_labels.float())
                    # print result every 1000 iteration during validation
                    if idx==0 or idx % round(1000/args.batch_size)==round(1000/args.batch_size)-1:
                        # ipdb.set_trace()
                        image = Image.open(os.path.join(data_const.original_image_dir, img_name[0][:].astype(np.uint8).tostring().decode('ascii'))).convert('RGB')
                        image_temp = image.copy()
                        raw_outputs = nn.Sigmoid()(outputs[0:int(edge_num[0])])
                        raw_outputs = raw_outputs.cpu().detach().numpy()
                        # class_img = vis_img(image, det_boxes, roi_labels, roi_scores)
                        class_img = vis_img_vcoco(image, det_boxes[0], roi_labels[0], roi_scores[0], edge_labels[0:int(edge_num[0])].cpu().numpy(), score_thresh=0.7)
                        action_img = vis_img_vcoco(image_temp, det_boxes[0], roi_labels[0], roi_scores[0], raw_outputs, score_thresh=0.5)
                        writer.add_image('gt_detection', np.array(class_img).transpose(2,0,1))
                        writer.add_image('action_detection', np.array(action_img).transpose(2,0,1))
                        writer.add_text('img_name', img_name[0][:].astype(np.uint8).tostring().decode('ascii'), epoch)

                idx+=1
                # accumulate loss of each batch
                running_loss += loss.item() * edge_labels.shape[0]
                # all_edge += edge_labels.shape[0]
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataset[phase])
            # epoch_loss = running_loss / all_edge
            # import ipdb; ipdb.set_trace()
            # log trainval datas, and visualize them in the same graph
            if phase == 'train':
                train_loss = epoch_loss 
                VcocoDataset.displaycount() 
            else:
                writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Loss: {} Execution time: {}".format(\
                        phase, epoch+1, args.epoch, epoch_loss, (end_time-start_time)))
                        
        # scheduler.step()
        # save model
        if epoch_loss<0.0405 or epoch % args.save_every == (args.save_every - 1) and epoch >= (500-1):
            checkpoint = { 
                            'lr': args.lr,
                           'b_s': args.batch_size,
                          'bias': args.bias, 
                            'bn': args.bn, 
                       'dropout': args.drop_prob,
                        'layers': args.layers,
                     'feat_type': args.feat_type,
                    'multi_head': args.multi_attn,
                     'diff_edge': args.diff_edge,
                    'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

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

parser = argparse.ArgumentParser(description="HOI DETECTION!")

parser.add_argument('--batch_size', '--b_s', type=int, default=32,required=True,
                    help='batch size: 32')
parser.add_argument('--layers', type=int, default=1, required=True,
                    help='the num of gcn layers: 1') 
parser.add_argument('--drop_prob', type=float, default=0.5, required=True,
                    help='dropout parameter: 0.5')
parser.add_argument('--lr', type=float, default=0.00001, required=True,
                    help='learning rate: 0.001')
parser.add_argument('--gpu', type=str2bool, default='true', 
                    help='chose to use gpu or not: True') 
parser.add_argument('--bias', type=str2bool, default='true', required=True,
                    help="add bias to fc layers or not: True")
parser.add_argument('--bn', type=str2bool, default='false', 
                    help='use batch normailzation or not: true')
# parse.add_argument('--bn', action="store_true", default=False,
#                     help='visualize the result or not')
parser.add_argument('--multi_attn', '--m_a', type=str2bool, default='false', required=True,
                     help='use multi attention or not: False')
parser.add_argument('--data_aug', '--d_a', type=str2bool, default='false', required=True,
                    help='data argument: false')

parser.add_argument('--img_data', type=str, default='datasets/hico/images/train2015',
                    help='location of the original dataset')
parser.add_argument('--pretrained', '-p', type=str, default=None,
                    help='location of the pretrained model file for training: None')
parser.add_argument('--log_dir', type=str, default='./log/vcoco',
                    help='path to save the log data like loss\accuracy... : ./log') 
parser.add_argument('--save_dir', type=str, default='./checkpoints/vcoco',
                    help='path to save the checkpoints: ./checkpoints/vcoco')

parser.add_argument('--epoch', type=int, default=600,
                    help='number of epochs to train: 600') 
parser.add_argument('--start_epoch', type=int, default=0,
                    help='number of beginning epochs : 0') 
parser.add_argument('--print_every', type=int, default=10,
                    help='number of steps for printing training and validation loss: 10') 
parser.add_argument('--save_every', type=int, default=10,
                    help='number of steps for saving the model parameters: 50')                      

parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', required=True,
                    help='the version of code, will create subdir in log/ && checkpoints/ ')

parser.add_argument('--train_model', '--t_m', type=str, default='epoch', required=True,
                    choices=['epoch', 'iteration'],
                    help='the version of code, will create subdir in log/ && checkpoints/ ')

parser.add_argument('--feat_type', '--f_t', type=str, default='fc7', required=True, choices=['fc7', 'pool'],
                    help='if using graph head, here should be \'pool\': default(fc7) ')

parser.add_argument('--optim',  type=str, default='adam', choices=['sgd', 'adam'], required=True,
                    help='which optimizer to be use: adam ')

parser.add_argument('--diff_edge',  type=str2bool, default='false', required=True,
                    help='h_h edge, h_o edge, o_o edge are different with each other')

parser.add_argument('--sampler',  type=float, default=0, 
                    help='h_h edge, h_o edge, o_o edge are different with each other')

parser.add_argument('--hico',  type=str, default=None,
                    help='location of the pretrained model of HICO_DET dataset: None')

args = parser.parse_args() 

if __name__ == "__main__":
    data_const = VcocoConstants(feat_type=args.feat_type)
    run_model(args, data_const)
