import dgl
import torch
import torch.nn as nn
import torchvision
import numpy as np

# from model.utils import Predictor
from model.graph_head import TowMLPHead, ResBlockHead
from model.grnn import GRNN
from model.vcoco_config import CONFIGURATION
from model.utils import MLP
import ipdb

class NodeUpdate(nn.Module):
    def __init__(self, CONFIG):
        super(NodeUpdate, self).__init__()
        self.fc = MLP(CONFIG.G_N_L_S_U, CONFIG.G_N_A_U, CONFIG.G_N_B_U, CONFIG.G_N_BN_U, CONFIG.G_N_D_U)
        self.fc_lang = MLP(CONFIG.G_N_L_S2_U, CONFIG.G_N_A2_U, CONFIG.G_N_B2_U, CONFIG.G_N_BN2_U, CONFIG.G_N_D2_U)

    def forward(self, node):
        feat = torch.cat([node.data['n_f_original'], node.data['new_n_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec_original'], node.data['new_n_f_lang']], dim=1)
        n_feat = self.fc(feat)
        n_feat_lang = self.fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}

class Predictor(nn.Module):
    def __init__(self, CONFIG, HICO=None):
        super(Predictor, self).__init__()
        if not HICO:
            self.classifier = MLP(CONFIG.G_ER_L_S, CONFIG.G_ER_A, CONFIG.G_ER_B, CONFIG.G_ER_BN, CONFIG.G_ER_D)
        else:
            self.classifier = MLP(CONFIG.G_ER_L_S_HICO, CONFIG.G_ER_A_HICO, CONFIG.G_ER_B_HICO, CONFIG.G_ER_BN_HICO, CONFIG.G_ER_D_HICO)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge):
        feat = torch.cat([edge.dst['new_n_f'], edge.dst['new_n_f_lang'], edge.data['s_f'], edge.src['new_n_f_lang'], edge.src['new_n_f']], dim=1)
        # feat = torch.cat([edge.dst['new_n_f'], edge.dst['new_n_f_lang'], edge.dst['z_f_sp'], edge.data['s_f'], edge.src['new_n_f_lang'], edge.src['new_n_f'], edge.src['z_f_sp']], dim=1)
        pred = self.classifier(feat)
        # if the criterion is BCELoss, you need to uncomment the following code
        # output = self.sigmoid(output)
        return {'pred': pred}

class AGRNN(nn.Module):
    def __init__(self, feat_type='fc7', bias=True, bn=True, dropout=None, multi_attn=False, layer=1, diff_edge=False, HICO=None):
        super(AGRNN, self).__init__()
 
        self.multi_attn = multi_attn
        self.layer = layer
        self.diff_edge = diff_edge
        self.CONFIG1 = CONFIGURATION(feat_type=feat_type, layer=1, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)
        self.CONFIG2 = CONFIGURATION(feat_type=feat_type, layer=2, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)
        self.CONFIG3 = CONFIGURATION(feat_type=feat_type, layer=3, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)

        if not feat_type=='fc7':
            self.graph_head = TowMLPHead(self.CONFIG1.G_H_L_S, self.CONFIG1.G_H_A, self.CONFIG1.G_H_B, self.CONFIG1.G_H_BN, self.CONFIG1.G_H_D)

        self.grnn1 = GRNN(self.CONFIG1, multi_attn=multi_attn, diff_edge=diff_edge)
        if layer==2:
            self.grnn2 = GRNN(self.CONFIG1, multi_attn=False, diff_edge=diff_edge)
        if layer==3:
            self.grnn2 = GRNN(self.CONFIG1, multi_attn=False, diff_edge=diff_edge)
            self.grnn3 = GRNN(self.CONFIG1, multi_attn=False, diff_edge=diff_edge)

        if layer>1:
            self.h_node_update = NodeUpdate(self.CONFIG1)
            if diff_edge:
                self.o_node_update = NodeUpdate(self.CONFIG1)

        if HICO:
            self.edge_readout = Predictor(self.CONFIG1, HICO)
        else:
            self.edge_readout = Predictor(self.CONFIG1)

    def _build_graph(self, node_num, roi_label, node_space, diff_edge):

        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)

        edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._collect_edge(node_num, roi_label, node_space, diff_edge)
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph bi-directional

        return graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def _collect_edge(self, node_num, roi_label, node_space, diff_edge):

        # get human nodes && object nodes
        h_node_list = np.where(roi_label == 1)[0]   # !NOTE: the type of roi_label must be numpy.array
        obj_node_list = np.where(roi_label != 1)[0]

        edge_list = []
        h_h_e_list = []
        o_o_e_list = []
        h_o_e_list = []
        readout_edge_list = []
        readout_h_h_e_list = []
        readout_h_o_e_list = []
        # get all edge in the fully-connected graph
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                else:
                    edge_list.append((src, dst))
        # 
        if diff_edge:
            # get h_h edges && h_o edges && o_o edges
            for src in h_node_list:
                for dst in h_node_list:
                    if src == dst: continue
                    h_h_e_list.append((src, dst))
            
            for src in obj_node_list:
                for dst in obj_node_list:
                    if src == dst: continue
                    o_o_e_list.append((src, dst))

            h_o_e_list = [x for x in edge_list if x not in h_h_e_list+o_o_e_list]

        # get corresponding readout edge in the graph
        for dst in h_node_list:
            for src in range(node_num):
            # for src in range(len(h_node_list), node_num):
                if dst == src:
                    continue
                readout_edge_list.append((src, dst))
        # src_box_list = np.arange(roi_label.shape[0])
        # for dst in h_node_list:
        #     if dst == roi_label.shape[0]-1:
        #         continue
        #     src_box_list = src_box_list[1:]
        #     for src in src_box_list:
        #         readout_edge_list.append((src, dst))

        # # get corresponding readout h_h edges && h_o edges
        # temp_h_node_list = h_node_list[:]
        # for dst in h_node_list:
        #     if dst == h_node_list.shape[0]-1:
        #         continue
        #     temp_h_node_list = temp_h_node_list[1:]
        #     for src in temp_h_node_list:
        #         if src == dst: continue
        #         readout_h_h_e_list.append((src, dst))

        # readout_h_o_e_list = [x for x in readout_edge_list if x not in readout_h_h_e_list]

        # ipdb.set_trace()
        # add node space to match the batch graph
        h_node_list = (np.array(h_node_list)+node_space).tolist()
        obj_node_list = (np.array(obj_node_list)+node_space).tolist()
        h_h_e_list = (np.array(h_h_e_list)+node_space).tolist()
        o_o_e_list = (np.array(o_o_e_list)+node_space).tolist()
        h_o_e_list = (np.array(h_o_e_list)+node_space).tolist()

        readout_h_h_e_list = (np.array(readout_h_h_e_list)+node_space).tolist()
        readout_h_o_e_list = (np.array(readout_h_o_e_list)+node_space).tolist()   
        readout_edge_list = (np.array(readout_edge_list)+node_space).tolist()

        return edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def forward(self, node_num=None, feat=None, spatial_feat=None, word2vec=None, roi_label=None, validation=False, choose_nodes=None, remove_nodes=None):
        # set up graph
        batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, batch_readout_edge_list, batch_readout_h_h_e_list, batch_readout_h_o_e_list = [], [], [], [], [], [], [], [], []
        node_num_cum = np.cumsum(node_num) # !IMPORTANT
        for i in range(len(node_num)):
            # set node space
            node_space = 0
            if i != 0:
                node_space = node_num_cum[i-1]
            graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._build_graph(node_num[i], roi_label[i], node_space, diff_edge=self.diff_edge)
            # updata batch graph,
            batch_graph.append(graph)
            batch_h_node_list += h_node_list
            batch_obj_node_list += obj_node_list
            batch_h_h_e_list += h_h_e_list
            batch_o_o_e_list += o_o_e_list
            batch_h_o_e_list += h_o_e_list
            batch_readout_edge_list += readout_edge_list
            batch_readout_h_h_e_list += readout_h_h_e_list
            batch_readout_h_o_e_list += readout_h_o_e_list
        batch_graph = dgl.batch(batch_graph)

        # ipdb.set_trace()
        if not self.CONFIG1.feat_type == 'fc7':
            feat = self.graph_head(feat)

        # pass throuh gnn/gcn
        if self.layer==1:
            self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation, initial_feat=True)
            # batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))
            batch_graph.apply_edges(self.edge_readout, tuple(zip(*batch_readout_edge_list)))
        
        elif self.layer==2:
            feat, feat_lang = self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation, pop_feat=True, initial_feat=True)
            self.grnn2(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, feat_lang, validation)
            if self.diff_edge:
                # update node feature at the last layer 
                if not len(batch_h_node_list) == 0:
                    batch_graph.apply_nodes(self.h_node_update, batch_h_node_list)
                if not len(batch_obj_node_list) == 0:
                    batch_graph.apply_nodes(self.o_node_update, batch_obj_node_list)
            else:
                batch_graph.apply_nodes(self.h_node_update, batch_h_node_list+batch_obj_node_list)
            batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))
        
        else:
            feat, feat_lang = self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation, pop_feat=True, initial_feat=True)
            feat, feat_lang = self.grnn2(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, feat_lang, validation, pop_feat=True)
            self.grnn3(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, feat_lang, validation)
            if self.diff_edge:
                # update node feature at the last layer 
                if not len(batch_h_node_list) == 0:
                    batch_graph.apply_nodes(self.h_node_update, batch_h_node_list)
                if not len(batch_obj_node_list) == 0:
                    batch_graph.apply_nodes(self.o_node_update, batch_obj_node_list)
            else:
                batch_graph.apply_nodes(self.h_node_update, batch_h_node_list+batch_obj_node_list)
            batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))

        # import ipdb; ipdb.set_trace()
        if self.training or validation:
            # return batch_graph.edges[tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list)))].data['pred']
            # !NOTE: cannot use "batch_readout_h_o_e_list+batch_readout_h_h_e_list" because of the wrong order
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred']
        else:
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha_lang'] 

if __name__ == "__main__":
    model = AGRNN()

    node_num = 3
    edge_list = []
    for src in range(node_num):
        for dst in range(node_num):
            edge_list.append((src,dst))
    src, dst = tuple(zip(*edge_list))
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    g.add_edges(src, dst)
    import ipdb; ipdb.set_trace()
    e_data = torch.eye(9)
    n_data = torch.arange(9)
    g.edata['feat'] = e_data
    g.ndata['x'] = n_data
