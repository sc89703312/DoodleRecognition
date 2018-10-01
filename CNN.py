import torch.nn as nn
import torch.nn.functional as F
from DoodleRecognition.PreProcess import *

class Net(nn.Module):
    def __init__(self, pretrained_model, feature_extract):
        super(Net,self).__init__()
        ## feature_extract: True -> only train the FC Layer params
        set_parameter_requires_grad(pretrained_model, feature_extract)
        self.pretrained_model = pretrained_model
        self.base = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(512, len(classes))

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.linear(f)
        return y
