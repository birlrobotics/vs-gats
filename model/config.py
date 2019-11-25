'''
MODEL CONFIGURATION
'''

class CONFIGURATION(object):
    def __init__(self, feat_type='fc7', layer=1, bias=True, bn=False, dropout=0.2, multi_attn=False):
        
        self.feat_type = feat_type
        self.ACTION_NUM = 117
        # graph head model
        self.G_H_L_S = [12544, 2048, 1024]   # 
        self.G_H_A   = ['ReLU', 'ReLU']
        self.G_H_B   = bias
        self.G_H_BN  = bn
        self.G_H_D   = dropout

        # if multi_attn:
        if True:
            if feat_type=='fc7':
                if layer==1:
                    # readout
                    self.G_ER_L_S = [1024+300+16+300+1024, 1024, 117]
                    self.G_ER_A   = ['ReLU', 'Identity']
                    self.G_ER_B   = bias
                    self.G_ER_BN  = bn
                    self.G_ER_D   = dropout
                    # self.G_ER_GRU = 1024

                    # # gnn node function
                    self.G_N_L_S = [1024+1024, 1024]
                    self.G_N_A   = ['ReLU']
                    self.G_N_B   = bias
                    self.G_N_BN  = bn
                    self.G_N_D   = dropout
                    # self.G_N_GRU = 1024

                    # # gnn node function for language
                    self.G_N_L_S2 = [300+300, 300]
                    self.G_N_A2   = ['ReLU']
                    self.G_N_B2   = bias
                    self.G_N_BN2  = bn
                    self.G_N_D2   = dropout
                    # self.G_N_GRU2 = 1024

                    # gnn edge function1
                    self.G_E_L_S = [1024*2+16, 1024]
                    self.G_E_A   = ['ReLU']
                    self.G_E_B   = bias
                    self.G_E_BN  = bn
                    self.G_E_D   = dropout

                    # gnn edge function2 for language
                    self.G_E_L_S2 = [300*2, 1024]
                    self.G_E_A2   = ['ReLU']
                    self.G_E_B2   = bias
                    self.G_E_BN2  = bn
                    self.G_E_D2   = dropout

                    # gnn attention mechanism
                    self.G_A_L_S = [1024, 1]
                    self.G_A_A   = ['LeakyReLU']
                    self.G_A_B   = bias
                    self.G_A_BN  = bn
                    self.G_A_D   = dropout

                    # gnn attention mechanism2 for language
                    self.G_A_L_S2 = [1024, 1]
                    self.G_A_A2   = ['LeakyReLU']
                    self.G_A_B2   = bias
                    self.G_A_BN2  = bn
                    self.G_A_D2   = dropout

            else:
                if layer==1:
                    # # gnn node function
                    self.G_N_L_S = [3072, 1024]
                    self.G_N_A   = ['ReLU']
                    self.G_N_B   = bias
                    self.G_N_BN  = bn
                    self.G_N_D   = dropout
                    self.G_N_GRU = 1024

                    # gnn edge function1
                    self.G_E_L_S = [1024*2, 1024]
                    self.G_E_A   = ['ReLU']
                    self.G_E_B   = bias
                    self.G_E_BN  = bn
                    self.G_E_D   = dropout

                    # gnn edge function2
                    self.G_E_L_S2 = [176, 512, 1024]
                    self.G_E_A2   = ['ReLU', 'ReLU']
                    self.G_E_B2   = bias
                    self.G_E_BN2  = bn
                    self.G_E_D2   = dropout

                    # gnn attention mechanism
                    self.G_A_L_S = [1024, 1]
                    self.G_A_A   = ['LeakyReLU']
                    self.G_A_B   = bias
                    self.G_A_BN  = bn
                    self.G_A_D   = dropout

                    # gnn attention mechanism2
                    self.G_A_L_S2 = [1024, 1]
                    self.G_A_A2   = ['LeakyReLU']
                    self.G_A_B2   = bias
                    self.G_A_BN2  = bn
                    self.G_A_D2   = dropout

        else:
            if feat_type=='fc7':
                if layer==1:
                    # # gnn node function
                    self.G_N_L_S = [1024*2, 1024]
                    self.G_N_A   = ['ReLU']
                    self.G_N_B   = bias
                    self.G_N_BN  = bn
                    self.G_N_D   = dropout
                    self.G_N_GRU = 1024

                    # gnn edge function1
                    self.G_E_L_S = [1024*2, 1024]
                    self.G_E_A   = ['ReLU']
                    self.G_E_B   = bias
                    self.G_E_BN  = bn
                    self.G_E_D   = dropout

                    # gnn attention mechanism
                    self.G_A_L_S = [1024, 1]
                    self.G_A_A   = ['LeakyReLU']
                    self.G_A_B   = False #bias
                    self.G_A_BN  = False #bn
                    self.G_A_D   = False #dropout
            else:
                if layer==1:
                    # gnn node function
                    self.G_N_L_S = [2048*2, 2048, 1024]
                    self.G_N_A   = ['ReLU', 'ReLU']
                    self.G_N_B   = bias
                    self.G_N_BN  = bn
                    self.G_N_D   = dropout
                    self.G_N_GRU = 1024

                    # gnn edge function
                    self.G_E_L_S = [2048*2, 512]
                    self.G_E_A   = ['ReLU']
                    self.G_E_B   = bias
                    self.G_E_BN  = bn
                    self.G_E_D   = dropout

                    # gnn attention mechanism
                    self.G_A_L_S = [512, 1]
                    self.G_A_A   = ['LeakyReLU']
                    self.G_A_B   = bias
                    self.G_A_BN  = bn
                    self.G_A_D   = dropout

        # if multi_attn:
        #     if feat_type=='fc7':
        #         if layer==1:
        #             # # gnn node function
        #             self.G_N_L_S = [3072, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function1
        #             self.G_E_L_S = [1024*2, 1024]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn edge function2
        #             self.G_E_L_S2 = [616, 512, 1024]
        #             self.G_E_A2   = ['ReLU', 'ReLU']
        #             self.G_E_B2   = bias
        #             self.G_E_BN2  = bn
        #             self.G_E_D2   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [1024, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout

        #             # gnn attention mechanism2
        #             self.G_A_L_S2 = [1024, 1]
        #             self.G_A_A2   = ['ReLU', 'LeakyReLU']
        #             self.G_A_B2   = bias
        #             self.G_A_BN2  = bn
        #             self.G_A_D2   = dropout

        #         elif layer==2:
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024, 1024]
        #             self.G_N_A   = ['ReLU','ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         else :
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #     else:
        #         if layer==1:
        #             # gnn node function
        #             self.G_N_L_S = [2048*2, 2048, 1024]
        #             self.G_N_A   = ['ReLU', 'ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [2048*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         elif layer==2:
        #             # gnn node function
        #             self.G_N_L_S = [1024*2, 1024, 1024]
        #             self.G_N_A   = ['ReLU', 'ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         else :
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        # else:
        #     if feat_type=='fc7':
        #         if layer==1:
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function1
        #             self.G_E_L_S = [1024*2, 1024]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [1024, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout

        #         elif layer==2:
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024, 1024]
        #             self.G_N_A   = ['ReLU','ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         else :
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #     else:
        #         if layer==1:
        #             # gnn node function
        #             self.G_N_L_S = [2048*2, 2048, 1024]
        #             self.G_N_A   = ['ReLU', 'ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [2048*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         elif layer==2:
        #             # gnn node function
        #             self.G_N_L_S = [1024*2, 1024, 1024]
        #             self.G_N_A   = ['ReLU', 'ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout
        #         else :
        #             # # gnn node function
        #             self.G_N_L_S = [1024*2, 1024]
        #             self.G_N_A   = ['ReLU']
        #             self.G_N_B   = bias
        #             self.G_N_BN  = bn   
        #             self.G_N_D   = dropout
        #             self.G_N_GRU = 1024

        #             # gnn edge function
        #             self.G_E_L_S = [1024*2, 512]
        #             self.G_E_A   = ['ReLU']
        #             self.G_E_B   = bias
        #             self.G_E_BN  = bn
        #             self.G_E_D   = dropout

        #             # gnn attention mechanism
        #             self.G_A_L_S = [512, 1]
        #             self.G_A_A   = ['LeakyReLU']
        #             self.G_A_B   = bias
        #             self.G_A_BN  = bn
        #             self.G_A_D   = dropout

    def save_config(self):
        model_config = {'graph_head':{}, 
                        'graph_node':{},
                        'graph_edge':{},
                        'graph_attn':{}}
        CONFIG=self.__dict__
        for k, v in CONFIG.items():
            if 'G_H' in k:
                model_config['graph_head'][k]=v
            elif 'G_N' in k:
                model_config['graph_node'][k]=v
            elif 'G_E' in k:
                model_config['graph_edge'][k]=v
            elif 'G_A' in k:
                model_config['graph_attn'][k]=v
            else:
                model_config[k]=v
        
        return model_config

if __name__=="__main__":
    data_const = CONFIGURATION()
    data_const.save_config()

