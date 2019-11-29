import torch.nn as nn
import torch
from neur.module import OrthoLinear, SVDLinear, SVDLinearWithInverse
import torch.nn.functional as F


class ResBlock(nn.Module):
    '''
    This is the repeating unit of the network
    '''

    def __init__(self, width, bottleneck):
        super().__init__()

        self.fcn1 = nn.Linear(width, bottleneck)
        #nn.init.orthogonal_(self.fcn1.weight)
        #nn.init.zeros_(self.fcn1.bias)
        self.nonlin = nn.ReLU()
        self.fcn2 = nn.Linear(bottleneck, width)
        #nn.init.orthogonal_(self.fcn2.weight)
        #nn.init.zeros_(self.fcn2.bias)

    def forward(self, x):
        identity = x

        out = self.fcn1(x)
        out = self.nonlin(out)

        out = self.fcn2(out)
        out += identity
        return out

class OrthoResBlock(nn.Module):
    '''
    This is the repeating unit of the network
    '''

    def __init__(self, width):
        super().__init__()

        self.fcn1 = SVDLinearWithInverse(width,width)
        self.nonlin = nn.ReLU()
        self.fcn2 = SVDLinear(width,width)
        self.alpha = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        identity = x

        out = self.fcn1(x)
        #vals = out[:,0]
        #vals = self.nonlin(vals)

        #out=torch.cat((vals.reshape(-1,1),out[:,1].reshape(-1,1)),dim=1)
        #out = self.fcn2(out)

        out = out*self.alpha+identity
        return out


class ResSVD(nn.Module):
    '''
    This is the whole network - it's composed of a sequence of blocks.
    '''

    def __init__(self, depth, num_inputs, width, bottleneck):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(OrthoResBlock(width))

        self.first = OrthoLinear(width)

        self.blocks = torch.nn.Sequential(*layers)
        self.final = torch.nn.Linear(width, 1)
        torch.nn.init.constant_(self.final.weight,-1)
        torch.nn.init.zeros_(self.final.bias)

    def forward(self, x):
        out = self.first(x)
        out = self.blocks(out)
        out = out.sum(dim=1)
        return out

class Res(nn.Module):
    '''
    This is the whole network - it's composed of a sequence of blocks.
    '''

    def __init__(self, depth, num_inputs, width, bottleneck):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(ResBlock(width,bottleneck))

        self.first = nn.Linear(num_inputs,width)

        self.blocks = torch.nn.Sequential(*layers)
        self.final = torch.nn.Linear(width, 1)
        #torch.nn.init.constant_(self.final.weight,-1)
        #torch.nn.init.zeros_(self.final.bias)

    def forward(self, x):
        out = self.first(x)
        out = self.blocks(out)
        out = self.final(out)
        return out
