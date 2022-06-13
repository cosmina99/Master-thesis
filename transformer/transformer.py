import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import numpy as np


OUTPUT_DIM = {
    'vit_b_16'				:  768,
    'vit_l_16'              :  1024,
}

class Transformer(nn.Module):

    def __init__(self,architecture, pretrained_flag = True):

        super(Transformer, self).__init__()

        network = getattr(models,architecture)(pretrained=pretrained_flag)
        
        #the backbone is up until node getitem_5
        if architecture.startswith('vit'):
            self.backbone = create_feature_extractor(network, return_nodes=['getitem_5'])

        self.norm = F.normalize
        
        #network information
        self.meta = {
            'architecture' : architecture, 
            'pooling' : "gem",
            'mean' : [0.485, 0.456, 0.406], #imagenet statistics
            'std' : [0.229, 0.224, 0.225], #imagenet statistics
            'outputdim' : OUTPUT_DIM[architecture],
        }


    def forward(self, img):

        x = self.norm(self.backbone(img)['getitem_5'])

        return x

def extract_features(net,dataloader,ms=[1],msp=1):
    
    net.eval()
    with torch.no_grad():
        vecs = np.zeros((net.meta['outputdim'], len(dataloader.dataset)))
        for i,input in enumerate(dataloader):
            vecs[:, i] = extract(net, input[0].cuda(), ms, msp)
            print("image: "+str(i))

    return vecs.T

def extract(net, input, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:    
            input_t = nn.functional.interpolate(input, size = 224, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v
