'''
MODEL CONFIGURATION
'''

ACTION_NUM = 117

# graph head model
G_H_L_S = [12544, 2048, 2048]   # 
G_H_A   = ['ReLU', 'ReLU']
G_H_B   = True
G_H_BN  = False
G_H_D   = False

# gnn node function
G_N_L_S = [1024*2, 1024]
G_N_A   = ['ReLU']
G_N_B   = True
G_N_BN  = False
G_N_D   = False
G_N_GRU = 1024

# gnn edge function
G_E_L_S = [1024*2, 1024, 512]
G_E_A   = ['ReLU', 'ReLU']
G_E_B   = True
G_E_BN  = False
G_E_D   = False

# gnn attention mechanism
G_A_L_S = [512, 1]
G_A_A   = ['LeakyReLU']
G_A_B   = False
G_A_BN  = False
G_A_D   = False