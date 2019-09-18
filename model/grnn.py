import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import MLP, Predictor
import ipdb

class H_H_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG, multi_attn):
        super(H_H_EdgeApplyMoudle, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        if multi_attn:
            self.edge_fc2 = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)

        if self.multi_attn:
            feat2 = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
            e_feat2 = self.edge_fc2(feat2)
            
            return {'e_f': e_feat, 'e_f2': e_feat2}   
        return {'e_f': e_feat}

class O_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG, multi_attn):
        super(O_O_EdgeApplyMoudle, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        if multi_attn:
            self.edge_fc2 = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)

        if self.multi_attn:
            feat2 = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
            e_feat2 = self.edge_fc2(feat2)
            
            return {'e_f': e_feat, 'e_f2': e_feat2}   
        return {'e_f': e_feat}  

class H_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, CONFIG, multi_attn):
        super(H_O_EdgeApplyMoudle, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        if multi_attn:
            self.edge_fc2 = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)

        if self.multi_attn:
            feat2 = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
            e_feat2 = self.edge_fc2(feat2)
            
            return {'e_f': e_feat, 'e_f2': e_feat2}   
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

class E_AttentionModule1(nn.Module):
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        return {'a_feat': a_feat}

class E_AttentionModule2(nn.Module):
    def __init__(self, CONFIG):
        super(E_AttentionModule2, self).__init__()
        self.attn_fc2 = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        # feat = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
        # a_feat2 = self.attn_fc2(feat)
        a_feat2 = self.attn_fc2(edge.data['e_f2'])

        return {'a_feat2': a_feat2}

class GNN(nn.Module):
    def __init__(self, CONFIG, multi_attn=False):
        super(GNN, self).__init__()
        self.multi_attn = multi_attn
        self.apply_h_h_edge = H_H_EdgeApplyMoudle(CONFIG, multi_attn)
        self.apply_h_o_edge = H_O_EdgeApplyMoudle(CONFIG, multi_attn)
        self.apply_o_o_edge = O_O_EdgeApplyMoudle(CONFIG, multi_attn)
        self.apply_edge_attn1 = E_AttentionModule1(CONFIG)
        if multi_attn:
            self.apply_edge_attn2 = E_AttentionModule2(CONFIG)  
        self.apply_h_node = H_NodeApplyModule(CONFIG)
        self.apply_o_node = O_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        # ipdb.set_trace()
        if self.multi_attn:
            return {'nei_n_f': edges.src['n_f'], 'e_f2': edges.data['e_f2'], 'a_feat': edges.data['a_feat'], 'a_feat2': edges.data['a_feat2']}
        return {'nei_n_f': edges.src['n_f'],  'a_feat': edges.data['a_feat']}

    def _reduce_func(self, nodes):
        # calculate the features of virtual nodes 
        # ipdb.set_trace()
        alpha1 = F.softmax(nodes.mailbox['a_feat'], dim=1)
        if self.multi_attn:
            alpha2 = F.softmax(nodes.mailbox['a_feat2'], dim=1)
            alpha = (alpha1+alpha2)/2
            z_raw_f = torch.cat([nodes.mailbox['nei_n_f'], nodes.mailbox['e_f2']], dim=2)
        else:
            alpha = alpha1
            z_raw_f = nodes.mailbox['nei_n_f']
        z_f = torch.sum( alpha * z_raw_f, dim=1)
        # when training batch_graph, here will process batch_graph graph by graph, 
        # we cannot return 'alpha' for the different dimension 
        if self.training or validation:
            return {'z_f': z_f}
        else:
            return {'z_f': z_f, 'alpha': alpha}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_feat=False):
        
        if not len(h_h_e_list) == 0:
            g.apply_edges(self.apply_h_h_edge, tuple(zip(*h_h_e_list)))
        # ipdb.set_trace()
        if not len(o_o_e_list) == 0:
            g.apply_edges(self.apply_o_o_edge, tuple(zip(*o_o_e_list)))
        if not len(h_o_e_list) == 0:
            g.apply_edges(self.apply_h_o_edge, tuple(zip(*h_o_e_list)))

        g.apply_edges(self.apply_edge_attn1)
        if self.multi_attn:
            g.apply_edges(self.apply_edge_attn2)    

        g.update_all(self._message_func, self._reduce_func)

        if not len(h_node) == 0:
            g.apply_nodes(self.apply_h_node, h_node)
        if not len(o_node) == 0:
            g.apply_nodes(self.apply_o_node, o_node)

        # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE
        g.ndata.pop('n_f')
        g.ndata.pop('z_f')
        g.edata.pop('e_f')
        g.edata.pop('a_feat')
        if self.multi_attn:
            g.ndata.pop('word2vec')
            g.edata.pop('s_f')
            g.edata.pop('e_f2')
            g.edata.pop('a_feat2')

        if pop_feat:
            return g.ndata.pop('new_n_f')
        # # ipdb.set_trace()
        # if self.training or validation:
        #     return g.ndata.pop('pred')
        # else:
        #     return g.ndata.pop('pred'), g.ndata.pop('alpha')

class GRNN(nn.Module):
    def __init__(self, CONFIG, multi_attn):
        super(GRNN, self).__init__()
        self.multi_attn = multi_attn
        self.gnn = GNN(CONFIG,multi_attn)

    def forward(self, batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, valid=False, pop_feat=False):
        # !NOTE: if node_num==1, there will be something wrong to forward the attention mechanism
        # ipdb.set_trace()
        global validation 
        validation = valid

        # initialize the graph with some datas
        batch_graph.ndata['n_f'] = feat
        if self.multi_attn:
            batch_graph.ndata['word2vec'] = word2vec
            batch_graph.edata['s_f'] = spatial_feat

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
        