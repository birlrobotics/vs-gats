import dgl
import torch
import torch.nn as nn
import torchvision
import numpy as np

# from model.utils import Predictor
from model.graph_head import TowMLPHead, ResBlockHead
from model.s3d_g import S3D_G
from model.grnn import GRNN
from model.config import CONFIGURATION
import ipdb

class Predictor(nn.Module):
    def __init__(self, in_feat, num_calss):
        super(Predictor, self).__init__()
        self.classifier = nn.Linear(in_feat, num_calss)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node):
        pred = self.classifier(node.data['new_n_f'])
        # if the criterion is BCELoss, you need to uncomment the following code
        # output = self.sigmoid(output)
        return {'pred': pred}

class AGRNN(nn.Module):
    def __init__(self, feat_type='fc7', bias=True, bn=True, dropout=None, multi_attn=False):
        super(AGRNN, self).__init__()
        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # self.extractor = S3D_G(initial_temporal_size=32, in_channel=3, gate=True)
        self.multi_attn = multi_attn
        self.CONFIG1 = CONFIGURATION(feat_type=feat_type, layer=1, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)
        self.CONFIG2 = CONFIGURATION(feat_type=feat_type, layer=2, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)
        self.CONFIG3 = CONFIGURATION(feat_type=feat_type, layer=3, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)

        if not feat_type=='fc7':
            self.graph_head = TowMLPHead(self.CONFIG1.G_H_L_S, self.CONFIG1.G_H_A, self.CONFIG1.G_H_B, self.CONFIG1.G_H_BN, self.CONFIG1.G_H_D)

        self.grnn1 = GRNN(self.CONFIG1, multi_attn=multi_attn)
        # self.grnn2 = GRNN(self.CONFIG2)
        # self.grnn3 = GRNN(self.CONFIG3)

        self.h_node_readout = Predictor(self.CONFIG1.G_N_L_S[-1], self.CONFIG1.ACTION_NUM)
        self.o_node_readout = Predictor(self.CONFIG1.G_N_L_S[-1], self.CONFIG1.ACTION_NUM)

    def _build_graph(self, node_num, roi_label, node_space):

        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)

        edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list = self._collect_edge(node_num, roi_label, node_space)
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph bi-directional

        return graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list

    def _collect_edge(self, node_num, roi_label, node_space):
        # get all edge in the fully-connected graph
        edge_list = []
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                else:
                    edge_list.append((src, dst))
        # get human nodes && object nodes
        h_node_list = np.where(roi_label == 1)[0]
        obj_node_list = np.where(roi_label != 1)[0]

        # get h_h edges && h_o edges && o_o edges
        h_h_e_list = []
        for src in h_node_list:
            for dst in h_node_list:
                if src == dst: continue
                h_h_e_list.append((src, dst))
        o_o_e_list = []
        for src in obj_node_list:
            for dst in obj_node_list:
                if src == dst: continue
                o_o_e_list.append((src, dst))
        h_o_e_list = [x for x in edge_list if x not in h_h_e_list+o_o_e_list]

        # ipdb.set_trace()
        # add node space to match the batch graph
        h_node_list = (np.array(h_node_list)+node_space).tolist()
        obj_node_list = (np.array(obj_node_list)+node_space).tolist()
        h_h_e_list = (np.array(h_h_e_list)+node_space).tolist()
        o_o_e_list = (np.array(o_o_e_list)+node_space).tolist()
        h_o_e_list = (np.array(h_o_e_list)+node_space).tolist()

        return edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list

    def forward(self, node_num=None, feat=None, spatial_feat=None, word2vec=None, roi_label=None, validation=False, choose_nodes=None, remove_nodes=None):
        # set up graph
        batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list= [], [], [], [], [], []
        node_num_cum = np.cumsum(node_num) # !IMPORTANT
        for i in range(len(node_num)):
            # set node space
            node_space = 0
            if i != 0:
                node_space = node_num_cum[i-1]
            graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list = self._build_graph(node_num[i], roi_label[i], node_space)
            # updata batch graph
            batch_graph.append(graph)
            batch_h_node_list += h_node_list
            batch_obj_node_list += obj_node_list
            batch_h_h_e_list += h_h_e_list
            batch_o_o_e_list += o_o_e_list
            batch_h_o_e_list += h_o_e_list
        batch_graph = dgl.batch(batch_graph)

        # ipdb.set_trace()
        if not self.CONFIG1.feat_type == 'fc7':
            feat = self.graph_head(feat)

        # pass throuh gcn
        # feat = self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, validation, pop_feat=True)
        # feat = self.grnn2(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, validation, pop_feat=True)
        self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation)
        # apply READOUT function to get predictions
        if not len(batch_h_node_list) == 0:
            batch_graph.apply_nodes(self.h_node_readout, batch_h_node_list)
        if not len(batch_obj_node_list) == 0:
            batch_graph.apply_nodes(self.o_node_readout, batch_obj_node_list)

        if self.training or validation:
            return batch_graph.ndata.pop('pred')
        else:
            return batch_graph.ndata.pop('pred'), batch_graph.ndata.pop('alpha')

if __name__ == "__main__":
    model = AGRNN()



    # @staticmethod
    # def _build_graph(node_num, roi_label, node_space):

    #     graph = dgl.DGLGraph()
    #     graph.add_nodes(node_num)
        
    #     edge_list = []
    #     for src in range(node_num):
    #         for dst in range(node_num):
    #             if src == dst:
    #                 continue
    #             else:
    #                 edge_list.append((src, dst))
    #     src, dst = tuple(zip(*edge_list))
    #     graph.add_edges(src, dst)   # make the graph bi-directional

    #     # get human nodes && object nodes
    #     h_node_list = np.where(roi_label == 1)[0]
    #     obj_node_list = np.where(roi_label != 1)[0]

    #     # get h_h edges && h_o edges && o_o edges
    #     h_h_e_list = []
    #     for src in h_node_list:
    #         for dst in h_node_list:
    #             if src == dst: continue
    #             h_h_e_list.append((src, dst))
    #     o_o_e_list = []
    #     for src in obj_node_list:
    #         for dst in obj_node_list:
    #             if src == dst: continue
    #             o_o_e_list.append((src, dst))
    #     h_o_e_list = [x for x in edge_list if x not in h_h_e_list+o_o_e_list]

    #     # ipdb.set_trace()
    #     # add node space to match the batch graph
    #     h_node_list = (np.array(h_node_list)+node_space).tolist()
    #     obj_node_list = (np.array(obj_node_list)+node_space).tolist()
    #     h_h_e_list = (np.array(h_h_e_list)+node_space).tolist()
    #     o_o_e_list = (np.array(o_o_e_list)+node_space).tolist()
    #     h_o_e_list = (np.array(h_o_e_list)+node_space).tolist()

    #     return graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list