import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection
    before the last ReLU activation.
    """ 

    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        
        # Block before skip connection        
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias = False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias = False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        residual = x

        out = self.block(x)

        out += residual
        out = nn.ReLU(out)

        return out
    
    
class DualRes(nn.Module):
    def __init__(self, input_shape, num_filters, num_resblocks):
        super(DualRes, self).__init__()
        self.input_shape = input_shape
        
        # Convolutional Block
        self.convblock = nn.Sequential(
            nn.Conv2d(input_shape[0], num_filters, 3, padding=1, bias = False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual Blocks
        self.resblocks = nn.ModuleList([*[ResBlock(num_filters) for _ in range(num_resblocks)]])
        
        # Policy Head
        self.ph_features = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        
        self.ph_fc = nn.Sequential(
            nn.Linear(num_filters, 3),
            nn.Softmax()
        )
        
        # Value Head
        self.vh_features = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        self.vh_fc = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        # Calculate body
        x = self.convblock(x)
        for block in self.resblocks:
            x = block(x)
        
        # Calculate policy
        ph = self.ph_features(x)
        ph = ph.view(ph.size(0), -1)
        ph = self.ph_fc(ph)
        
        # Calculate value
        vh = self.vh_features(x)
        vh = vh.view(vh.size(0), -1)
        vh = self.vh_fc(vh)
        
        return (ph, vh)