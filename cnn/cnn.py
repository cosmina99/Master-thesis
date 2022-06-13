import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import numpy as np

OUTPUT_DIM = {
    'resnet18'              :  512,
    'resnet50'              :  2048,
    'efficientnet_b4'		:  1792,
}

class GeM(nn.Module):
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6):

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class CNN(nn.Module):

    def __init__(self,architecture,gem_p = 3,pretrained_flag = True):

        super(CNN, self).__init__()
        network = getattr(models,architecture)(pretrained=pretrained_flag) #load the selected model

        #the backbone is up until the convolutional layers
        if architecture.startswith('resnet'):
            self.backbone = nn.Sequential(*list(network.children())[:-2])
        elif architecture.startswith('efficientnet'):
            self.backbone = nn.Sequential(*list(network.children())[:-1])

        self.pool = GeM(p = gem_p) #performing gem pooling
        self.norm = F.normalize #l2 normalisation

        #network info
        self.meta = {
            'architecture' : architecture, 
            'pooling' : "gem",
            'mean' : [0.485, 0.456, 0.406], #imagenet statistics
            'std' : [0.229, 0.224, 0.225], #imagenet statistics
            'outputdim' : OUTPUT_DIM[architecture],
        }

    def forward(self, img):

        x = self.norm(self.pool(self.backbone(img)))
        
        return x

def extract(net, input, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:    
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v

def extract_features(net, dataloader, ms=[1], msp=1):

    net.eval()
    with torch.no_grad():
        vecs = np.zeros((net.meta['outputdim'], len(dataloader.dataset)))
        for i,input in enumerate(dataloader):
            vecs[:, i] = extract(net,input[0].cuda(), ms, msp)
            print("image: " + str(i))

    return vecs.T
