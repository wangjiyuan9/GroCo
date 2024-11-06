from __future__ import absolute_import, division, print_function

import torch.nn as nn

from networks.monovit.hr_decoder import HR_DepthDecoder
from networks.monovit.mpvit import *


class DeepNet(nn.Module):
    def __init__(self, opts, weights_init="pretrained", num_layers=18, num_pose_frames=2, scales=range(4)):
        super(DeepNet, self).__init__()
        self.num_layers = num_layers
        self.weights_init = weights_init
        self.num_pose_frames = num_pose_frames
        self.scales = scales

        self.encoder = mpvit_small()
        self.decoder = HR_DepthDecoder(opts)


    def forward(self, inputs):
        self.outputs = self.decoder(self.encoder(inputs))
        return self.outputs
