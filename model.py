# -*- coding:utf-8 -*-
import torchvision.models as models
import ref as ref
import torch
import torch.nn as nn
import os
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
from models.network import PyramidHourglassNet

#重新初始化optimizer
def getModel(opt): 
    if 'hg' in opt.arch:
        model = PyramidHourglassNet(opt.nStack, opt.nModules, opt.nFeats, opt.numOutput)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.LR, alpha = ref.alpha,
                                                                    eps = ref.epsilon,
                                                                    weight_decay = ref.weightDecay,
                                                                    momentum = ref.momentum)

    if opt.loadModel != 'none':
        checkpoint = torch.load(opt.loadModel)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()    
        model.load_state_dict(state_dict)
        
    return model, optimizer
    
def saveModel(model, optimizer, path):
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
