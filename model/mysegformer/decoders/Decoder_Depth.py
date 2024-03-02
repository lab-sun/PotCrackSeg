import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim


        # RGB decoder

        self.linear_c4_rgb = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3_rgb = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2_rgb = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1_rgb = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse_rgb = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred_rgb = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)


        #depth decoder

        self.linear_c4_depth = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3_depth = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2_depth = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1_depth = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse_depth = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred_depth = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, rgb,depth):

        verbose = False
        c1_d, c2_d, c3_d, c4_d = depth

        if verbose: print("c1_d size",c1_d.size())
        if verbose: print("c2_d size",c2_d.size())
        if verbose: print("c3_d size",c3_d.size())
        if verbose: print("c4_d size",c4_d.size())
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_d.shape

        _c4_d = self.linear_c4_depth(c4_d).permute(0,2,1).reshape(n, -1, c4_d.shape[2], c4_d.shape[3])
        if verbose: print("_c4_d size after linear_c4_depth",_c4_d.size())

        _c4_d = F.interpolate(_c4_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c4_d size after interpolate",_c4_d.size())

        _c3_d = self.linear_c3_depth(c3_d).permute(0,2,1).reshape(n, -1, c3_d.shape[2], c3_d.shape[3])
        if verbose: print("_c3_d size after linear_c3_depth",_c3_d.size())

        _c3_d = F.interpolate(_c3_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c3_d size after interpolate",_c3_d.size())

        _c2_d = self.linear_c2_depth(c2_d).permute(0,2,1).reshape(n, -1, c2_d.shape[2], c2_d.shape[3])
        if verbose: print("_c2_d size after linear_c2_depth",_c2_d.size())

        _c2_d = F.interpolate(_c2_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c2_d size after interpolate",_c2_d.size())

        _c1_d = self.linear_c1_depth(c1_d).permute(0,2,1).reshape(n, -1, c1_d.shape[2], c1_d.shape[3])
        if verbose: print("_c1_d size after linear_c1_depth",_c1_d.size())

        _c_d = self.linear_fuse_depth(torch.cat([_c4_d, _c3_d, _c2_d, _c1_d], dim=1))
        x_d = self.dropout(_c_d)
        x_d = self.linear_pred_depth(x_d)

        return x_d

        
