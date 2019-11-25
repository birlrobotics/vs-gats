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

class COMBINE_EdgeApplyModule(nn.Module):
    def __init__(self, CONFIG, multi_attn=False):
        super(COMBINE_EdgeApplyModule, self).__init__()
        self.multi_attn = multi_attn
        self.edge_fc = MLP(CONFIG.G_E_L_S_C, CONFIG.G_E_A_C, CONFIG.G_E_B_C, CONFIG.G_E_BN_C, CONFIG.G_E_D_C)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['combined_feat'], edge.data['s_f'], edge.dst['combined_feat']], dim=1)

        c_e_feat = self.edge_fc(feat)
  
        return {'c_e_f': c_e_feat}

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
        combined_feat = torch.cat([n_feat, n_feat_lang], dim=1)

        return {'combined_feat': combined_feat}

class COMBINE_NodeApplyModule(nn.Module):
    def __init__(self, CONFIG):
        super(COMBINE_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S_C, CONFIG.G_N_A_C, CONFIG.G_N_B_C, CONFIG.G_N_BN_C, CONFIG.G_N_D_C)

    
    def forward(self, node):
        feat = torch.cat([node.data['combined_feat'], node.data['c_z_f']], dim=1)
        n_feat = self.node_fc(feat)

        return {'c_new_n_f': n_feat}

class E_AttentionModule1(nn.Module):
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)
        self.attn_fc_lang = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        # ipdb.set_trace()
        a_feat = self.attn_fc(edge.data['e_f'])
        a_feat_lang = self.attn_fc_lang(edge.data['e_f_lang'])
        return {'a_feat': a_feat, 'a_feat_lang': a_feat_lang}

class COMBINE_AttentionModule2(nn.Module):
    def __init__(self, CONFIG):
        super(COMBINE_AttentionModule2, self).__init__()
        self.attn_fc2 = MLP(CONFIG.G_A_L_S_C, CONFIG.G_A_A_C, CONFIG.G_A_B_C, CONFIG.G_A_BN_C, CONFIG.G_A_D_C)

    def forward(self, edge):
        # feat = torch.cat([edge.src['word2vec'], edge.data['s_f'], edge.dst['word2vec']], dim=1)
        # a_feat2 = self.attn_fc2(feat)
        c_a_feat = self.attn_fc2(edge.data['c_e_f'])

        return {'c_a_feat': c_a_feat}

class GNN(nn.Module):
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True):
        super(GNN, self).__init__()

        self.multi_attn = multi_attn
        self.diff_edge = diff_edge
        self.apply_h_h_edge = H_H_EdgeApplyModule(CONFIG, multi_attn)
        self.apply_edge_attn1 = E_AttentionModule1(CONFIG)  
        self.apply_h_node = H_NodeApplyModule(CONFIG)
        self.apply_combine_edge = COMBINE_EdgeApplyModule(CONFIG)
        self.apply_COMBINE_attn = COMBINE_AttentionModule2(CONFIG)
        self.apply_combine_node = COMBINE_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        # ipdb.set_trace()
        if self.multi_attn:
            return {'nei_n_f': edges.src['n_f'], 'e_f2': edges.data['e_f2'], 'a_feat': edges.data['a_feat'], 'a_feat2': edges.data['a_feat2']}
        return {'nei_n_f': edges.src['n_f'], 'nei_n_w': edges.src['word2vec'], 'e_f': edges.data['e_f'], 'e_f_lang': edges.data['e_f_lang'], 'a_feat': edges.data['a_feat'], 'a_feat_lang': edges.data['a_feat_lang']}

    def _c_message_func(self, edges):
        return {'c_nei_n_f': edges.src['combined_feat'], 'c_e_f': edges.data['c_e_f'], 'c_a_feat': edges.data['c_a_feat']}

    def _c_reduce_func(self, nodes):
        c_alpha = F.softmax(nodes.mailbox['c_a_feat'], dim=1)
        c_z_raw_f = nodes.mailbox['c_nei_n_f']+nodes.mailbox['c_e_f']
        c_z_f = torch.sum( c_alpha * c_z_raw_f, dim=1)

        return {'c_z_f': c_z_f}

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
            # return {'z_f': z_f, 'z_f_lang': z_f_lang, 'z_f_sp': z_f_sp}
        else:
            return {'z_f': z_f, 'z_f_lang': z_f_lang, 'alpha': alpha, 'alpha_lang': alpha_lang}
            # return {'z_f': z_f, 'z_f_lang': z_f_lang, 'z_f_sp': z_f_sp, 'alpha': alpha, 'alpha_lang': alpha_lang}

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
            # ipdb.set_trace()
            # g.apply_edges(self.apply_h_h_edge, tuple(zip(*(h_h_e_list+h_o_e_list+o_o_e_list))))
            g.apply_edges(self.apply_h_h_edge, g.edges())
            g.apply_edges(self.apply_edge_attn1)
            g.update_all(self._message_func, self._reduce_func)
            g.apply_nodes(self.apply_h_node, h_node+o_node)
            g.apply_edges(self.apply_combine_edge, g.edges())
            g.apply_edges(self.apply_COMBINE_attn)
            g.update_all(self._c_message_func, self._c_reduce_func)
            g.apply_nodes(self.apply_combine_node, h_node+o_node)

        # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE

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
            # if self.training or validation:
            #     output = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
            #     return output
            # else:
            #     output, alpha = self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
            #     return output, alpha
        except Exception as e:
            print(e)
            ipdb.set_trace()
        