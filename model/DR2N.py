import torch
import torch.nn as nn
import torchvision
import dgl
import networkx as nx

from s3d_g import S3D_G
from grnn import GRNN
import ipdb

class DR2N(nn.Module):
    def __init__(self, ):
        super(DR2N, self).__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.extractor = S3D_G(initial_temporal_size=32, in_channel=3, gate=True)
        self.grnn = GRNN(2 * 1024, 1024)

    def forward(self, image, frames, action_label, per_t):
        # get region porposals and labels
        rois, roi_labels, _ = self.detector(image)
        # get features cropping out inside the bounding box
        features = self.extractor(frames, rois)
        # setup the graph 
        node_num = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)
        edge_list = []
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    break
                else:
                    edge_list.append((src, dst))
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph undirectional
        # pass grnn 
        output = self.grnn(graph, features, roi_labels, action_label)

        return output

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from PIL import Image
    from data_tools.dataset import Dataset
    testdata = Dataset(data_dir='dataset/VOCdevkit2007/VOC2007')
    image = Image.open('dataset/demo.jpg')
    image.show()
    model = DR2N(21)
    model.eval()
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    output = model([image_tensor])
    print(output)