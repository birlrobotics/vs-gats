import torch
import torch.nn as nn
import torchvision

from model.graph_head import TowMLPHead, ResBlockHead
from model.s3d_g import S3D_G
from model.grnn import GRNN
from model.config import CONFIGURATION
import ipdb

class AGRNN(nn.Module):
    def __init__(self, feat_type='fc7'):
        super(AGRNN, self).__init__()
        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # self.extractor = S3D_G(initial_temporal_size=32, in_channel=3, gate=True)
        # self.grnn = GRNN(2 * 1024, 1024)
        self.CONFIG = CONFIGURATION(feat_type=feat_type)
        if not feat_type=='fc7':
            self.graph_head = TowMLPHead(self.CONFIG.G_H_L_S, self.CONFIG.G_H_A, self.CONFIG.G_H_B, self.CONFIG.G_H_BN, self.CONFIG.G_H_D)
        self.grnn = GRNN(self.CONFIG)

    def forward(self, node_num, feat, roi_label, validation=False):
        # ipdb.set_trace()
        if not self.CONFIG.feat_type == 'fc7':
            feat = self.graph_head(feat)
        if self.training or validation:
            output = self.grnn(node_num, feat, roi_label, validation)
            return output
        else:
            output, alpha = self.grnn(node_num, feat, roi_label)
            return output, alpha

if __name__ == "__main__":
    model = AGRNN()