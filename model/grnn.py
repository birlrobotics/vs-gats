import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import ipdb
from model.utils import MLP, Predictor

class H_H_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG):
        super(H_H_EdgeApplyMoudle, self).__init__()
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)
        
        return {'e_f': e_feat}    

class O_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG):
        super(O_O_EdgeApplyMoudle, self).__init__()
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
    
    def forward(self, edge):
        # ipdb.set_trace()
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)

        return {'e_f': e_feat}  

class H_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG):
        super(H_O_EdgeApplyMoudle, self).__init__()
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        # self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_E_BN, CONFIG.G_A_D)
    
    def forward(self, edge):
        # ipdb.set_trace()
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)
        # a_feat = self.attn_fc(e_feat)
        # # alpha = F.softmax(a_feat, dim=1)
        # return {'e_f': e_feat, 'a_feat': a_feat} 
        return {'e_f': e_feat}    

class H_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(H_NodeApplyModule, self).__init__()
        # self.node_fn = nn.Linear(in_feat, out_feat)
        # if activation == 'rule':
        #     self.activation = nn.ReLU()
        # if activation == 'sigmoid':
        #     self.activation = nn.Sigmoid()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.gru = nn.GRU(CONFIG.G_N_L_S[-1], CONFIG.G_N_GRU)
        self.predictor = Predictor(CONFIG.G_N_GRU, CONFIG.ACTION_NUM)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)  
        # ipdb.set_trace()
        n_feat = self.node_fc(feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class O_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(O_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.gru = nn.GRU(CONFIG.G_N_L_S[-1], CONFIG.G_N_GRU)
        self.predictor = Predictor(CONFIG.G_N_GRU, CONFIG.ACTION_NUM)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fc(feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class E_AttentionModule(nn.Module):
    def __init__(self, CONFIG):
        super(E_AttentionModule, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_E_BN, CONFIG.G_A_D)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        return {'a_feat': a_feat}

class GNN(nn.Module):
    def __init__(self, CONFIG):
        super(GNN, self).__init__()
        self.apply_h_h_edge = H_H_EdgeApplyMoudle(CONFIG)
        self.apply_h_o_edge = H_O_EdgeApplyMoudle(CONFIG)
        self.apply_o_o_edge = O_O_EdgeApplyMoudle(CONFIG)
        self.apply_edge_attn= E_AttentionModule(CONFIG)
        self.apply_h_node = H_NodeApplyModule(CONFIG)
        self.apply_o_node = O_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        # ipdb.set_trace()
        return {'nei_n_f': edges.src['n_f'],  'a_feat': edges.data['a_feat']}

    def _reduce_func(self, nodes):
        # calculate the features of virtual nodes 
        # ipdb.set_trace()
        alpha = F.softmax(nodes.mailbox['a_feat'], dim=1)
        # z = torch.sum(torch.mul(alpha.repeat(1,1,1024), nodes.mailbox['nei_n_f']), dim=1).squeeze()
        z = torch.sum( alpha * nodes.mailbox['nei_n_f'], dim=1)
        return {'z_f': z, 'alpha': alpha}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list):
        # ipdb.set_trace()
        if not len(h_h_e_list) == 0:
            g.apply_edges(self.apply_h_h_edge, tuple(zip(*h_h_e_list)))

        # ipdb.set_trace()
        if not len(o_o_e_list) == 0:
            g.apply_edges(self.apply_o_o_edge, tuple(zip(*o_o_e_list)))
        if not len(h_o_e_list) == 0:
            g.apply_edges(self.apply_h_o_edge, tuple(zip(*h_o_e_list)))

        g.apply_edges(self.apply_edge_attn)
        g.update_all(self._message_func, self._reduce_func)

        if not len(h_node) == 0:
            g.apply_nodes(self.apply_h_node, h_node)
        if not len(o_node) == 0:
            g.apply_nodes(self.apply_o_node, o_node)

        # ipdb.set_trace()
        return g.ndata.pop('pred'), g.ndata.pop('alpha')

class GRNN(nn.Module):
    def __init__(self, CONFIG):
        super(GRNN, self).__init__()
        self.gnn = GNN(CONFIG)

    def forward(self, node_num, node_feat, roi_label):
        # set up graph
        # ipdb.set_trace()
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)
        # !NOTE: if node_num==1, there is something wrong to forward the attention mechanism
        if node_num == 1:
            print("just one node. no edges")     
        else:
            edge_list = []
            for src in range(node_num):
                for dst in range(node_num):
                    if src == dst:
                        continue
                    else:
                        edge_list.append((src, dst))
            src, dst = tuple(zip(*edge_list))
            graph.add_edges(src, dst)   # make the graph bi-directional

        # get human nodes && object nodes
        h_node = np.where(roi_label == 1)
        obj_node = np.where(roi_label != 1)

        # if h_node[0].shape[0] == 2:
        #     ipdb.set_trace()
        # get h_h edges && h_o edges && o_o edges
        h_h_e_list = []
        for src in h_node[0]:
            for dst in h_node[0]:
                if src == dst: continue
                h_h_e_list.append((src, dst))
        o_o_e_list = []
        for src in obj_node[0]:
            for dst in obj_node[0]:
                if src == dst: continue
                o_o_e_list.append((src, dst))
        h_o_e_list = [x for x in edge_list if x not in h_h_e_list+o_o_e_list]

        graph.ndata['n_f'] = node_feat
        try:
            output, alpha = self.gnn(graph, h_node[0], obj_node[0], h_h_e_list, o_o_e_list, h_o_e_list)
        except Exception as e:
            print(e)
            ipdb.set_trace()
        return output, alpha