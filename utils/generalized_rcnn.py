# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import copy
import ipdb

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None, save_feat=False):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        #ipdb.set_trace()
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, save_feat=save_feat)
        if save_feat:
            detections[0]['boxes'] = detections[0]['boxes'].reshape(-1, 4)
            detection = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            detection[0]['boxes'] = detection[0]['boxes'].reshape(-1, 81*4)
            return detection
        else:
            bbox_wst_processed_img = detections[0]['boxes'][:]
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            if self.training:
                return losses

            return [bbox_wst_processed_img, detections, features, images.image_sizes]
