import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import ipdb

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x

def get_activation(name):
    if name=='ReLU':
        return nn.ReLU(inplace=True)
    elif name=='Tanh':
        return nn.Tanh()
    elif name=='Identity':
        return Identity()
    elif name=='Sigmoid':
        return nn.Sigmoid()
    elif name=='LeakyReLU':
        return nn.LeakyReLU(0.2,inplace=True)
    else:
        assert(False), 'Not Implemented'

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation, bias=True, use_bn=True, drop_prob=None):
        '''
        Args:
             layer_sizes: a list, the size of each layer you want to construct: [1024,1024,...]
              activation: a list, the activations of each layer you want to use: ['ReLU', 'Tanh',...]
                  use_bn: bool, use batch normalize or not
               drop_prob: default is None, use drop out layer or not
        '''
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias)
            activate = get_activation(activation[i])
            block = nn.Sequential(
                OrderedDict([(f'L{i}', layer), 
                             (f'A{i}', activate)
                            ]))
            if use_bn:
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                block.add_module(f'B{i}', bn)
            if drop_prob is not None:
                block.add_module(f'D{i}', nn.Dropout(drop_prob))
            self.layers.append(block)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

class H_H_EdgeApplyMoudle(nn.Module):
    def __init__(self, feat_sizes, atten_layers, edge_activation, atten_activation, bias=True, use_bn=True, drop_prob=None):
        super(H_H_EdgeApplyMoudle, self).__init__()
        self.edge_fc = MLP(layer_sizes=feat_sizes, activation=edge_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
        self.attn_fc = MLP(layer_sizes=feat_sizes[-1:]*atten_layers+[1], activation=atten_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)
        a_feat = self.attn_fc(e_feat)
        # alpha = F.softmax(a_feat, dim=1)

        return {'e_f': e_feat, 'alpha': alpha}   

class O_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, feat_sizes, atten_layers, edge_activation, atten_activation, bias=True, use_bn=True, drop_prob=None):
        super(O_O_EdgeApplyMoudle, self).__init__()
        self.edge_fc = MLP(layer_sizes=feat_sizes, activation=edge_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
        self.attn_fc = MLP(layer_sizes=feat_sizes[-1:]*atten_layers+[1], activation=atten_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)
        a_feat = self.attn_fc(e_feat)
        # alpha = F.softmax(a_feat, dim=1)

        return {'e_f': e_feat, 'alpha': alpha}    

class H_O_EdgeApplyMoudle(nn.Module):
    def __init__(self, feat_sizes, atten_layers, edge_activation, atten_activation, bias=True, use_bn=True, drop_prob=None):
        super(H_O_EdgeApplyMoudle, self).__init__()

        self.edge_fc = MLP(layer_sizes=feat_sizes, activation=edge_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
        self.attn_fc = MLP(layer_sizes=feat_sizes[-1:]*atten_layers+[1], activation=atten_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
    
    def forward(self, edge):
        # ipdb.set_trace()
        feat = torch.cat([edge.src['n_f'], edge.dst['n_f']], dim=1)
        e_feat = self.edge_fc(feat)
        a_feat = self.attn_fc(e_feat)
        # alpha = F.softmax(a_feat, dim=1)

        return {'e_f': e_feat, 'a_feat': a_feat}     

class H_NodeApplyModule(nn.Module):
    def __init__(self, feat_sizes, hidden_size, action_num, node_activation, bias=True, use_bn=True, drop_prob=None):
        super(H_NodeApplyModule, self).__init__()
        # self.node_fn = nn.Linear(in_feat, out_feat)
        # if activation == 'rule':
        #     self.activation = nn.ReLU()
        # if activation == 'sigmoid':
        #     self.activation = nn.Sigmoid()
        self.node_fc = MLP(layer_sizes=feat_sizes, activation=node_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
        self.gru = nn.GRU(feat_sizes[-1], hidden_size)
        self.predictor = Predictor(hidden_size, action_num)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)  
        # ipdb.set_trace()
        n_feat = self.node_fc(feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class O_NodeApplyModule(nn.Module):
    def __init__(self, feat_sizes, hidden_size, action_num, node_activation, bias=True, use_bn=True, drop_prob=None):
        super(O_NodeApplyModule, self).__init__()
        self.node_fc = MLP(layer_sizes=feat_sizes, activation=node_activation, bias=bias, use_bn=use_bn, drop_prob=drop_prob)
        self.gru = nn.GRU(feat_sizes[-1], hidden_size)
        self.predictor = Predictor(hidden_size, action_num)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        n_feat = self.node_fc(feat)
        pred = self.predictor(n_feat)
        return {'pred': pred}

class GNN(nn.Module):
    def __init__(self, feat_sizes, atten_layers, hidden_size, action_num, node_activation, edge_activation, atten_activation, bias, use_bn, drop_prob):
        super(GNN, self).__init__()
        self.apply_h_h_edge = H_H_EdgeApplyMoudle(feat_sizes, atten_layers, edge_activation, atten_activation, bias, use_bn, drop_prob)
        self.apply_h_o_edge = H_O_EdgeApplyMoudle(feat_sizes, atten_layers, edge_activation, atten_activation, bias, use_bn, drop_prob)
        self.apply_o_o_edge = O_O_EdgeApplyMoudle(feat_sizes, atten_layers, edge_activation, atten_activation, bias, use_bn, drop_prob)
        self.apply_h_node = H_NodeApplyModule(feat_sizes, hidden_size, action_num, node_activation, bias, use_bn, drop_prob)
        self.apply_o_node = O_NodeApplyModule(feat_sizes, hidden_size, action_num, node_activation, bias, use_bn, drop_prob)

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

    def forward(self, g, h_node, o_node):
        # ipdb.set_trace()
        g.apply_edges(self.apply_h_o_edge)
        g.update_all(self._message_func, self._reduce_func)
        g.apply_nodes(self.apply_h_node, h_node)
        g.apply_nodes(self.apply_o_node, o_node)
        # ipdb.set_trace()
        return g.ndata.pop('pred'), g.ndata.pop('alpha')


class GRNN(nn.Module):
    def __init__(self, feat_sizes=[2*1024, 1024, 1024], atten_layers=2, hidden_size=1024, action_num=117, \
                 node_activation=['ReLU']*2, edge_activation=['ReLU']*2, atten_activation=['LeakyReLU']*2, bias=True, use_bn=True, drop_prob=None):
        super(GRNN, self).__init__()
        self.gnn = GNN(feat_sizes, atten_layers, hidden_size, action_num, node_activation, edge_activation, atten_activation, bias, use_bn, drop_prob)

    def forward(self, node_num, node_feat, roi_label):
        # set up graph
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
        h_node = np.where(roi_label == 1 )
        obj_node = np.where(roi_label != 1)
        graph.ndata['n_f'] = node_feat
        try:
            output, alpha = self.gnn(graph, h_node[0], obj_node[0])
        except Exception as e:
            print(e)
            ipdb.set_trace()
        return output, alpha