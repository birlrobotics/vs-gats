import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import MLP, Predictor
import ipdb

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
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        # self.gru = nn.GRU(CONFIG.G_N_L_S[-1], CONFIG.G_N_GRU)
        # self.predictor = Predictor(CONFIG.G_N_GRU, CONFIG.ACTION_NUM)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fc(feat)
        # pred = self.predictor(n_feat)
        # return {'pred': pred}
        return {'new_n_f': n_feat}

class O_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(O_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        # self.gru = nn.GRU(CONFIG.G_N_L_S[-1], CONFIG.G_N_GRU)
        # self.predictor = Predictor(CONFIG.G_N_GRU, CONFIG.ACTION_NUM)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fc(feat)
        # pred = self.predictor(n_feat)
        # return {'pred': pred}
        return {'new_n_f': n_feat}

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
        # when training batch_graph, here will process batch_graph graph by graph, 
        # we cannot return 'alpha' for the different dimension 
        if self.training or validation:
            return {'z_f': z}
        else:
            return {'z_f': z, 'alpha': alpha}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_feat=False):
        
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

        if pop_feat:
            # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE
            g.ndata.pop('n_f')
            g.ndata.pop('z_f')
            g.edata.pop('a_feat')
            g.edata.pop('e_f')
            return g.ndata.pop('new_n_f')
        # # ipdb.set_trace()
        # if self.training or validation:
        #     return g.ndata.pop('pred')
        # else:
        #     return g.ndata.pop('pred'), g.ndata.pop('alpha')

class GRNN(nn.Module):
    def __init__(self, CONFIG):
        super(GRNN, self).__init__()
        self.gnn = GNN(CONFIG)

    def forward(self, batch_graph, node_feat, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, valid=False, pop_feat=False):
        # !NOTE: if node_num==1, there is something wrong to forward the attention mechanism
        # ipdb.set_trace()
        global validation 
        validation = valid

        # batch_graph = batch_graph[0]
        batch_graph.ndata['n_f'] = node_feat
        try:
            if pop_feat:
                feat = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, pop_feat=pop_feat)
                return feat
            else:
                self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, pop_feat=pop_feat)
            # if self.training or validation:
            #     output = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
            #     return output
            # else:
            #     output, alpha = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
            #     return output, alpha
        except Exception as e:
            print(e)
            ipdb.set_trace()
        