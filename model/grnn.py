import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class H_H_EdgeApplyMoudle(nn.Module):
    def __init__(self, in_feat, out_feat, activation):
        super(H_H_EdgeApplyMoudle, self).__init__()
        self.edge_fn = nn.Linear(in_feat, out_feat, bias=False)
        if activation == 'rule':
            self.activation = nn.ReLU()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        self.attn_fc = nn.Linear(out_feat, 1, bias=False)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fn(feat)
        e_feat = self.activation(e_feat)
        # attention
        a_feat = self.attn_fc(e_feat)
        a_fact = F.leaky_relu(a_feat)
        alpha = F.softmax(a_fact, dim=1)

        return {'e_f': e_feat, 'alpha': alpha}   

class O_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, in_feat, out_feat, activation):
        super(O_O_EdgeApplyMoudle, self).__init__()
        self.edge_fn = nn.Linear(in_feat, out_feat, bias=False)
        if activation == 'rule':
            self.activation = nn.ReLU()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        self.attn_fc = nn.Linear(out_feat, 1, bias=False)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fn(feat)
        e_feat = self.activation(e_feat)
        # attention
        a_feat = self.attn_fc(e_feat)
        a_fact = F.leaky_relu(a_feat)
        alpha = F.softmax(a_fact, dim=1)

        return {'e_f': e_feat, 'alpha': alpha}    

class H_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, in_feat, out_feat, activation):
        super(H_O_EdgeApplyMoudle, self).__init__()
        self.edge_fn = nn.Linear(in_feat, out_feat, bias=False)
        if activation == 'rule':
            self.activation = nn.ReLU()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        self.attn_fc = nn.Linear(out_feat, 1, bias=False)
    
    def forward(self, edge):
        # ipdb.set_trace()
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fn(feat)
        e_feat = self.activation(e_feat)
        # attention
        a_fact = self.attn_fc(e_feat)
        a_fact = F.leaky_relu(a_fact)

        return {'e_f': e_feat, 'a_fact': a_fact}     

class H_NodeApplyModule(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_size, action_num, activation):
        super(H_NodeApplyModule, self).__init__()
        self.node_fn = nn.Linear(in_feat, out_feat)
        if activation == 'rule':
            self.activation = nn.ReLU()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        self.gru = nn.GRU(out_feat, hidden_size)
        self.predictor = Predictor(hidden_size, action_num)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fn(feat)
        n_feat = self.activation(n_feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class O_NodeApplyModule(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_size, action_num, activation):
        super(O_NodeApplyModule, self).__init__()
        self.node_fn = nn.Linear(in_feat, out_feat)
        if activation == 'rule':
            self.activation = nn.ReLU()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        self.gru = nn.GRU(out_feat, hidden_size)
        self.predictor = Predictor(hidden_size, action_num)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fn(feat)
        n_feat = self.activation(n_feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class GNN(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_size, action_num, activation='rule'):
        super(GNN, self).__init__()
        self.apply_h_h_edge = H_H_EdgeApplyMoudle(in_feat, out_feat, activation=activation)
        self.apply_h_o_edge = H_O_EdgeApplyMoudle(in_feat, out_feat, activation=activation)
        self.apply_o_o_edge = O_O_EdgeApplyMoudle(in_feat, out_feat, activation=activation)
        self.apply_h_node = H_NodeApplyModule(in_feat, out_feat, hidden_size, action_num, activation=activation)
        self.apply_o_node = O_NodeApplyModule(in_feat, out_feat, hidden_size, action_num, activation=activation)

    def _message_func(self, edges):
        # ipdb.set_trace()
        return {'nei_n_f': edges.src['n_f'],  'a_fact': edges.data['a_fact']}

    def _reduce_func(self, nodes):
        # calculate the features of virtual nodes 
        # ipdb.set_trace()
        alpha = F.softmax(nodes.mailbox['a_fact'], dim=1)
        # z = torch.sum(torch.mul(alpha.repeat(1,1,1024), nodes.mailbox['nei_n_f']), dim=1).squeeze()
        z = torch.sum( alpha * nodes.mailbox['nei_n_f'], dim=1)
        return {'z_f': z, 'alpha': alpha}

    def forward(self, g, h_node, o_node):
        # ipdb.set_trace()
        g.apply_edges(self.apply_h_o_edge)
        g.update_all(self._message_func, self._reduce_func)
        g.apply_nodes(self.apply_h_node, h_node)
        g.apply_nodes(self.apply_o_node, o_node)
        # ipdb.set_trace()
        return g.ndata.pop('pred'), g.ndata.pop('alpha')

# construct the classifier
class Predictor(nn.Module):
    def __init__(self, in_feat, num_calss):
        super(Predictor, self).__init__()
        self.classifier = nn.Linear(in_feat, num_calss)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_f):
        output = self.classifier(h_f)
        # if the criterion is BCELoss, you need to uncomment the following code
        # output = self.sigmoid(output)
        return output

class GRNN(nn.Module):
    def __init__(self, in_feat=2*1024, out_feat=1024, hidden_size=1024, action_num=117):
        super(GRNN, self).__init__()
        self.gnn = GNN(in_feat=in_feat, out_feat=out_feat, hidden_size=hidden_size , action_num=117, activation='rule')

    def forward(self, g, node_feat, roi_label):
        # ipdb.set_trace()
        h_node = np.where(roi_label == 50 )
        obj_node = np.where(roi_label != 50)
        g.ndata['n_f'] = node_feat
        output, alpha = self.gnn(g, h_node[0], obj_node[0])
        # ipdb.set_trace()
        return output, alpha