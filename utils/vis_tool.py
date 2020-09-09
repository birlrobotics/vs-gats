import time
import ipdb

import numpy as np
import matplotlib
import torch as t
import random
# import visdom
# import cv2

matplotlib.use('Agg')
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw, ImageFont
from datasets import metadata, vcoco_metadata

def vis_img(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8, data_gt=False):
    try:
        if len(bboxs) == 0:
            return img    

        font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=25)
        line_width = 3
        if data_gt:
            Drawer = ImageDraw.Draw(img)
            r_color = random.choice(np.arange(256))
            g_color = random.choice(np.arange(256))
            b_color = random.choice(np.arange(256))

            Drawer.rectangle(list(bboxs[0]), outline=(120,0,0), width=line_width)
            Drawer.rectangle(list(bboxs[1]), outline=(120,0,0), width=line_width)
            im_w,im_h = img.size
            x1,y1,x2,y2 = bboxs[0]
            x1_,y1_,x2_,y2_ = bboxs[1]
            # import ipdb; ipdb.set_trace()
            c_xh = int(0.5*x1)+int(0.5*x2)
            c_yh = int(0.5*y1)+int(0.5*y2)
            c_xo = int(0.5*x1_)+int(0.5*x2_)
            c_yo = int(0.5*y1_)+int(0.5*y2_)
            c_xh = max(0,min(c_xh,im_w-1))
            c_yh = max(0,min(c_yh,im_h-1))
            c_xo = max(0,min(c_xo,im_w-1))
            c_yo = max(0,min(c_yo,im_h-1))
            
            Drawer.line(((c_xo,c_yo),(c_xh,c_yh)), fill=(r_color,g_color,b_color), width=3)
            # print(c_xo,c_yo,c_xh,c_yh)
                    
            
            text =  metadata.action_classes[raw_action]
            h, w = font.getsize(text)
            Drawer.rectangle(xy=(bboxs[0][0], bboxs[0][1], bboxs[0][0]+h+1, bboxs[0][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
            Drawer.text(xy=(bboxs[0][0], bboxs[0][1]), text=text, font=font, fill=None)

            return img

        human_num = len(np.where(labels == 1)[0])
        node_num = len(labels)
        labeled_edge_list = np.cumsum(node_num - np.arange(human_num) -1)
        labeled_edge_list[-1] = 0

        Drawer = ImageDraw.Draw(img)

        count = 0
        for h_idx in range(human_num):
            for i_idx in range(node_num):
                if i_idx <= h_idx:
                    continue
                edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
                action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]

                text = str()
                det_label, det_score = str(), str()
                if len(action_idx) > 0:
                    
                    r_color = random.choice(np.arange(256))
                    g_color = random.choice(np.arange(256))
                    b_color = random.choice(np.arange(256))

                    text1 = metadata.coco_classes[labels[i_idx]]
                    if text1 == 'pizza':
                        continue
                    # if text1=='cake': continue
                    Drawer.rectangle(list(bboxs[h_idx]), outline='#FF0000', width=line_width)
                    Drawer.rectangle(list(bboxs[i_idx]), outline='#FF0000', width=line_width)

                    h, w = font.getsize(text1)
                    # Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1], bboxs[i_idx][0]+h+1, bboxs[i_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]), text=text1, font=font, fill=None)
                    Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill=None)
                    im_w,im_h = img.size
                    x1,y1,x2,y2 = bboxs[h_idx]
                    x1_,y1_,x2_,y2_ = bboxs[i_idx]

                    c0 = int(0.5*x1)+int(0.5*x2)
                    r0 = int(0.5*y1)+int(0.5*y2)
                    c1 = int(0.5*x1_)+int(0.5*x2_)
                    r1 = int(0.5*y1_)+int(0.5*y2_)
                    c0 = max(0,min(c0,im_w-1))
                    c1 = max(0,min(c1,im_w-1))
                    r0 = max(0,min(r0,im_h-1))
                    r1 = max(0,min(r1,im_h-1))
                    # import ipdb; ipdb.set_trace()
                    Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=3)
                    
                    shift = 0
                    for i in range(len(action_idx)):
                        # if metadata.action_classes[action_idx[i]] == 'no_interaction':
                        #     continue
                        # text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                        # h, w = font.getsize(text)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                        det_label = det_label + " " + metadata.action_classes[action_idx[i]]
                        # det_score = det_score + "  " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                        det_score = det_score + "  " + str(round(raw_action[edge_idx][action_idx[i]],2))
                        h1, w1 = font.getsize(det_label)
                        h2, w2 = font.getsize(det_score)
                        Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h2+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-1), text=det_label, font=font, fill=None)
                    Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]+w1+1), text=det_score, font=font, fill=None)

                    # # up the bbox
                    #     Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1), text=det_label, font=font, fill=None)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-1), text=det_score, font=font, fill=None)

                    # # down the bbox
                    #     Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)

                    # if count == 0 :
                    #     count +=1
                    #     # up the bbox
                    #     for i in range(len(action_idx)):
                    #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                    #     #     h, w = font.getsize(text)
                    #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                    #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
                    #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                    #         h1, w1 = font.getsize(det_label)
                    #         h2, w2 = font.getsize(det_score)
                    #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #         Drawer.rectangle(xy=(bboxs[h_idx][0]-100, bboxs[h_idx][1]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     Drawer.text(xy=(bboxs[h_idx][0]-100, bboxs[h_idx][1]-w1-1), text=det_label, font=font, fill=None)
                    #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-1), text=det_score, font=font, fill=None)

                    # elif count == 1:
                    #     count+=1
                    #     # down the bbox
                    #     for i in range(len(action_idx)):
                    #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                    #     #     h, w = font.getsize(text)
                    #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                    #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
                    #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                    #         h1, w1 = font.getsize(det_label)
                    #         h2, w2 = font.getsize(det_score)
                    #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #         Drawer.rectangle(xy=(bboxs[h_idx][0]-0, bboxs[h_idx][3]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
                    #     Drawer.text(xy=(bboxs[h_idx][0]-0, bboxs[h_idx][3]-w1-1), text=det_label, font=font, fill=None)
                    #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)

                    # else:
                    #     count+=1
                    #     # down the bbox
                    #     for i in range(len(action_idx)):
                    #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                    #     #     h, w = font.getsize(text)
                    #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                    #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
                    #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                    #         h1, w1 = font.getsize(det_label)
                    #         h2, w2 = font.getsize(det_score)
                    #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #         Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
                    #     Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_label, font=font, fill=None)
                    #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)
                    
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

def vis_img_vcoco(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8, data_gt=False):
    try:
        if len(bboxs) == 0:
            return img    

        font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=25)
        line_width = 3
        if data_gt:
            Drawer = ImageDraw.Draw(img)
            r_color = random.choice(np.arange(256))
            g_color = random.choice(np.arange(256))
            b_color = random.choice(np.arange(256))

            Drawer.rectangle(list(bboxs[0]), outline=(120,0,0), width=line_width)
            Drawer.rectangle(list(bboxs[1]), outline=(120,0,0), width=line_width)
            im_w,im_h = img.size
            x1,y1,x2,y2 = bboxs[0]
            x1_,y1_,x2_,y2_ = bboxs[1]
            # import ipdb; ipdb.set_trace()
            c_xh = int(0.5*x1)+int(0.5*x2)
            c_yh = int(0.5*y1)+int(0.5*y2)
            c_xo = int(0.5*x1_)+int(0.5*x2_)
            c_yo = int(0.5*y1_)+int(0.5*y2_)
            c_xh = max(0,min(c_xh,im_w-1))
            c_yh = max(0,min(c_yh,im_h-1))
            c_xo = max(0,min(c_xo,im_w-1))
            c_yo = max(0,min(c_yo,im_h-1))
            
            Drawer.line(((c_xo,c_yo),(c_xh,c_yh)), fill=(r_color,g_color,b_color), width=3)
            # print(c_xo,c_yo,c_xh,c_yh)
                    
            
            text =  vcoco_metadata.action_class_with_object[raw_action]
            h, w = font.getsize(text)
            # Drawer.rectangle(xy=(bboxs[0][0], bboxs[0][1], bboxs[0][0]+h+1, bboxs[0][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
            # Drawer.text(xy=(bboxs[0][0], bboxs[0][1]), text=text, font=font, fill=None)
            Drawer.rectangle(xy=(bboxs[1][0], bboxs[1][1], bboxs[1][0]+h+1, bboxs[1][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
            Drawer.text(xy=(bboxs[1][0], bboxs[1][1]), text=text, font=font, fill=None)

            return img

        human_num = len(np.where(labels == 1)[0])
        node_num = len(labels)
        obj_num = node_num - human_num

        Drawer = ImageDraw.Draw(img)

        # count = 0
        for h_idx in range(human_num):
            same_human = 0
            for i_idx in range(node_num):
            # for i_idx in range(human_num, node_num):
                if h_idx == i_idx:
                    continue
                if h_idx == 0:
                    edge_idx = i_idx - 1
                elif h_idx > i_idx:
                    edge_idx = h_idx * (node_num-1) + i_idx
                else:
                    edge_idx = h_idx * (node_num-1) + i_idx -1
                # edge_idx = h_idx * obj_num + i_idx - human_num

                action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]
                action_idx = [id for id in action_idx if id != 0]

                text = str()
                det_label, det_score = str(), str()
                if len(action_idx) > 0:
                    
                    r_color = random.choice(np.arange(256))
                    g_color = random.choice(np.arange(256))
                    b_color = random.choice(np.arange(256))

                    text1 = vcoco_metadata.coco_classes[labels[i_idx]]

                    # if text1=='laptop': continue #FF0000

                    Drawer.rectangle(list(bboxs[h_idx]+[5,0,0,0]), outline='#FF8C00', width=line_width)
                    Drawer.rectangle(list(bboxs[i_idx]), outline='#FF8C00', width=line_width)

                    h, w = font.getsize(text1)
                    # Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1], bboxs[i_idx][0]+h+1, bboxs[i_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]), text=text1, font=font, fill=None)
                    Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill='#000000')
                    im_w,im_h = img.size
                    x1,y1,x2,y2 = bboxs[h_idx]
                    x1_,y1_,x2_,y2_ = bboxs[i_idx]

                    c0 = int(0.5*x1)+int(0.5*x2)
                    r0 = int(0.5*y1)+int(0.5*y2)
                    c1 = int(0.5*x1_)+int(0.5*x2_)
                    r1 = int(0.5*y1_)+int(0.5*y2_)
                    c0 = max(0,min(c0,im_w-1))
                    c1 = max(0,min(c1,im_w-1))
                    r0 = max(0,min(r0,im_h-1))
                    r1 = max(0,min(r1,im_h-1))
                    # import ipdb; ipdb.set_trace()
                    Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=3)
                    
                    shift = 0
                    same_human += 1
                    for i in range(len(action_idx)):
                    #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                    #     h, w = font.getsize(text)
                    #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                        # if vcoco_metadata.action_class_with_object[action_idx[i]] != 'jump':
                        #     continue
                        det_label = det_label + vcoco_metadata.action_class_with_object[action_idx[i]]
                        # det_score = det_score + "  " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                        det_score = det_score + "  " + str(round(raw_action[edge_idx][action_idx[i]],2))
                        # import ipdb; ipdb.set_trace()
                        h1, w1 = font.getsize(det_label)
                        h2, w2 = font.getsize(det_score)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1
                        # ]-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                        Drawer.rectangle(xy=(bboxs[h_idx][0]-h1, bboxs[h_idx][2]-270, bboxs[h_idx][0]+5, bboxs[h_idx][2]-270+w1+5), fill=(r_color,g_color,b_color), outline=None, width=0)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h2+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]), text=det_label, font=font, fill=None)
                    Drawer.text(xy=(bboxs[h_idx][0]-h1, bboxs[h_idx][2]-270), text=det_label, font=font, fill='#000000')
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]+w1+1), text=det_score, font=font, fill=None)
                    
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass


def vis_img_frcnn(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8):
    try:
        if len(bboxs) == 0:
            return img    

        if scores is not None:
            keep = np.where(scores > score_thresh)[0]
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
                text = metadata.coco_classes[label]
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

if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    from PIL import Image, ImageDraw
    writer = SummaryWriter(log_dir='log')
    img = cv2.imread('000001.jpg')
    img2 = Image.open('demo.jpg')
    ipdb.set_trace()
    img = img.transpose(2,0,1)
    cv2.rectangle(img, (20, 20), (220,220), (0,0,255), 5)
    drawer = ImageDraw.Draw(img2)
    while True:

        writer.add_image('test', img.transpose(2,0,1)[::-1])
        writer.add_image("test2", np.array(img2).transpose(2,0,1))
        time.sleep(5)
    writer.close()