import torch
import torch.nn as nn
import torchvision

from model.graph_head import TowMLPHead, ResBlockHead
from model.s3d_g import S3D_G
from model.grnn import GRNN
import model.config as CONFIG
import ipdb

class AGRNN(nn.Module):
    def __init__(self):
        super(AGRNN, self).__init__()
        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # self.extractor = S3D_G(initial_temporal_size=32, in_channel=3, gate=True)
        # self.grnn = GRNN(2 * 1024, 1024)
        self.graph_head = TowMLPHead(CONFIG.G_H_L_S, CONFIG.G_H_A, CONFIG.G_H_B, CONFIG.G_H_BN, CONFIG.G_H_D)
        self.grnn = GRNN(CONFIG)

    def forward(self, node_num, feat, roi_label, feat_type='fc7'):
        # ipdb.set_trace()
        if not feat_type == 'fc7':
            feat = self.graph_head(feat)
        output, alpha = self.grnn(node_num, feat, roi_label)

        return output, alpha

if __name__ == "__main__":
    model = AGRNN()