"""
Define nn.Sequential with multiple inputs and outputs, 
including backward function
"""
from torch import nn

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
    def back(self, *inputs):
        for module in self[::-1]._modules.values():
            if type(inputs) == tuple:
                inputs = module.back(*inputs)
            else:
                inputs = module.back(inputs)
        return inputs
    