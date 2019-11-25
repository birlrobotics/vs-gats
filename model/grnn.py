import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import MLP, Predictor
import ipdb

class H_H_EdgeApplyModule(nn.Module):
    def __init__(self, CONFIG, multi_attn=False):
        super(H_H_EdgeApplyModule, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        # feat_lang = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.dst['word2vec']], dim=1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}

class O_O_EdgeApplyModule(nn.Module):
    def __init__(self, CONFIG, multi_attn=False):
        super(O_O_EdgeApplyModule, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}

class H_O_EdgeApplyModule(nn.Module):
    def __init__(self, CONFIG, multi_attn=False):
        super(H_O_EdgeApplyModule, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        # feat_lang = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.dst['word2vec']], dim=1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}

class H_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(H_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.node_fc_lang = MLP(CONFIG.G_N_L_S2, CONFIG.G_N_A2, CONFIG.G_N_B2, CONFIG.G_N_BN2, CONFIG.G_N_D2)
    
    def forward(self, node):
        # import ipdb; ipdb.set_trace()
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec'], node.data['z_f_lang']], dim=1)
        n_feat = self.node_fc(feat)
        n_feat_lang = self.node_fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}

class O_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(O_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.node_fc_lang = MLP(CONFIG.G_N_L_S2, CONFIG.G_N_A2, CONFIG.G_N_B2, CONFIG.G_N_BN2, CONFIG.G_N_D2)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec'], node.data['z_f_lang']], dim=1)
        n_feat = self.node_fc(feat)
        n_feat_lang = self.node_fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}

class E_AttentionModule1(nn.Module):
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)
        self.attn_fc_lang = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        a_feat_lang = self.attn_fc_lang(edge.data['e_f_lang'])
        return {'a_feat': a_feat, 'a_feat_lang': a_feat_lang}

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
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True):
        super(GNN, self).__init__()

        self.multi_attn = multi_attn
        self.diff_edge = diff_edge
        self.apply_h_h_edge = H_H_EdgeApplyModule(CONFIG, multi_attn)
        self.apply_edge_attn1 = E_AttentionModule1(CONFIG)  
        self.apply_h_node = H_NodeApplyModule(CONFIG)
        if diff_edge:
            self.apply_h_o_edge = H_O_EdgeApplyModule(CONFIG, multi_attn)
            self.apply_o_o_edge = O_O_EdgeApplyModule(CONFIG, multi_attn)
            self.apply_o_node = O_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        # ipdb.set_trace()
        if self.multi_attn:
            return {'nei_n_f': edges.src['n_f'], 'e_f2': edges.data['e_f2'], 'a_feat': edges.data['a_feat'], 'a_feat2': edges.data['a_feat2']}
        return {'nei_n_f': edges.src['n_f'], 'nei_n_w': edges.src['word2vec'], 'e_f': edges.data['e_f'], 'e_f_lang': edges.data['e_f_lang'], 'a_feat': edges.data['a_feat'], 'a_feat_lang': edges.data['a_feat_lang']}

    def _reduce_func(self, nodes):
        # calculate the features of virtual nodes 
        # ipdb.set_trace()
        alpha = F.softmax(nodes.mailbox['a_feat'], dim=1)
        alpha_lang = F.softmax(nodes.mailbox['a_feat_lang'], dim=1)

        z_raw_f = nodes.mailbox['nei_n_f']+nodes.mailbox['e_f']
        # z_raw_f = nodes.mailbox['nei_n_f']
        z_f = torch.sum( alpha * z_raw_f, dim=1)

        z_raw_f_lang = nodes.mailbox['nei_n_w']
        z_f_lang = torch.sum(alpha_lang * z_raw_f_lang, dim=1)
        # when training batch_graph, here will process batch_graph graph by graph, 
        # we cannot return 'alpha' for the different dimension 
        if self.training or validation:
            return {'z_f': z_f, 'z_f_lang': z_f_lang}
        else:
            return {'z_f': z_f, 'z_f_lang': z_f_lang, 'alpha': alpha, 'alpha_lang': alpha_lang}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_feat=False):
        
        if self.diff_edge:
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

            # import ipdb; ipdb.set_trace()
            if not len(h_node) == 0:
                g.apply_nodes(self.apply_h_node, h_node)
            if not len(o_node) == 0:
                g.apply_nodes(self.apply_o_node, o_node)
        else:
            # g.apply_edges(self.apply_h_h_edge, tuple(zip(*(h_h_e_list+h_o_e_list+o_o_e_list))))
            g.apply_edges(self.apply_h_h_edge, g.edges())
            g.apply_edges(self.apply_edge_attn1)
            g.update_all(self._message_func, self._reduce_func)
            g.apply_nodes(self.apply_h_node, h_node+o_node)

        # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE
        g.ndata.pop('n_f')
        # g.edata.pop('s_f')
        g.ndata.pop('word2vec')

        g.ndata.pop('z_f')
        g.edata.pop('e_f')
        g.edata.pop('a_feat')

        g.ndata.pop('z_f_lang')
        g.edata.pop('e_f_lang')
        g.edata.pop('a_feat_lang')

        if pop_feat:
            return g.ndata.pop('new_n_f'), g.ndata.pop('new_n_f_lang')

class GRNN(nn.Module):
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True):
        super(GRNN, self).__init__()
        self.multi_attn = multi_attn
        self.gnn = GNN(CONFIG, multi_attn, diff_edge)

    def forward(self, batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, valid=False, pop_feat=False, initial_feat=False):
        # !NOTE: if node_num==1, there will be something wrong to forward the attention mechanism
        # ipdb.set_trace()
        global validation 
        validation = valid

        # initialize the graph with some datas
        batch_graph.ndata['n_f'] = feat
        batch_graph.ndata['word2vec'] = word2vec
        batch_graph.edata['s_f'] = spatial_feat
        if initial_feat:
            batch_graph.ndata['n_f_original'] = feat
            batch_graph.ndata['word2vec_original'] = word2vec

        try:
            if pop_feat:
                feat, feat_lang = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, pop_feat=pop_feat)
                return feat, feat_lang
            else:
                self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
        except Exception as e:
            print(e)
            ipdb.set_trace()
        