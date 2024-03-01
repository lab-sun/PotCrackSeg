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


        self.linear_fusion = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )

        #self.linear_pred_fusion = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)                



        self.DRC = DRC(embedding_dim,self.num_classes)
       
    def forward(self, rgb,depth):
        # len=4, 1/4,1/8,1/16,1/32

        c1_r, c2_r, c3_r, c4_r = rgb
        verbose = False

        if verbose: print("c1_r size",c1_r.size())
        if verbose: print("c2_r size",c2_r.size())
        if verbose: print("c3_r size",c3_r.size())
        if verbose: print("c4_r size",c4_r.size())

        c1_d, c2_d, c3_d, c4_d = depth

        if verbose: print("c1_d size",c1_d.size())
        if verbose: print("c2_d size",c2_d.size())
        if verbose: print("c3_d size",c3_d.size())
        if verbose: print("c4_d size",c4_d.size())
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_r.shape

        _c4_r = self.linear_c4_rgb(c4_r).permute(0,2,1).reshape(n, -1, c4_r.shape[2], c4_r.shape[3])
        if verbose: print("_c4_r size after linear_c4_rgb",_c4_r.size())
        _c4_d = self.linear_c4_depth(c4_d).permute(0,2,1).reshape(n, -1, c4_d.shape[2], c4_d.shape[3])
        if verbose: print("_c4_d size after linear_c4_depth",_c4_d.size())

        _c4_r = F.interpolate(_c4_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c4_r size after interpolate",_c4_r.size())
        _c4_d = F.interpolate(_c4_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c4_d size after interpolate",_c4_d.size())



        _c3_r = self.linear_c3_rgb(c3_r).permute(0,2,1).reshape(n, -1, c3_r.shape[2], c3_r.shape[3])
        if verbose: print("_c3_r size after linear_c3_rgb",_c3_r.size())
        _c3_d = self.linear_c3_depth(c3_d).permute(0,2,1).reshape(n, -1, c3_d.shape[2], c3_d.shape[3])
        if verbose: print("_c3_d size after linear_c3_depth",_c3_d.size())

        _c3_r = F.interpolate(_c3_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c3_r size after interpolate",_c3_r.size())
        _c3_d = F.interpolate(_c3_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c3_d size after interpolate",_c3_d.size())



        _c2_r = self.linear_c2_rgb(c2_r).permute(0,2,1).reshape(n, -1, c2_r.shape[2], c2_r.shape[3])
        if verbose: print("_c2_r size after linear_c2_rgb",_c2_r.size())
        _c2_d = self.linear_c2_depth(c2_d).permute(0,2,1).reshape(n, -1, c2_d.shape[2], c2_d.shape[3])
        if verbose: print("_c2_d size after linear_c2_depth",_c2_d.size())

        _c2_r = F.interpolate(_c2_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c2_r size after interpolate",_c2_r.size())
        _c2_d = F.interpolate(_c2_d, size=c1_d.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c2_d size after interpolate",_c2_d.size())



        _c1_d = self.linear_c1_depth(c1_d).permute(0,2,1).reshape(n, -1, c1_d.shape[2], c1_d.shape[3])
        if verbose: print("_c1_d size after linear_c1_depth",_c1_d.size())
        _c1_r = self.linear_c1_rgb(c1_r).permute(0,2,1).reshape(n, -1, c1_r.shape[2], c1_r.shape[3])
        if verbose: print("_c1_r size after linear_c1_rgb",_c1_r.size())


        _c_d = self.linear_fuse_depth(torch.cat([_c4_d, _c3_d, _c2_d, _c1_d], dim=1))
        x_d = self.dropout(_c_d)
        x_d = self.linear_pred_depth(x_d)

        _c_r = self.linear_fuse_rgb(torch.cat([_c4_r, _c3_r, _c2_r, _c1_r], dim=1))
        x_r = self.dropout(_c_r)
        x_r = self.linear_pred_rgb(x_r)


        fusion = self.linear_fusion(torch.cat([_c4_r+_c4_d, _c3_r+_c3_d, _c2_r+_c2_d, _c1_r+_c1_d], dim=1))

        

        rgb_comple,depth_comple,rgb_fusion,depth_fusion = self.DRC(fusion,x_r,x_d)


        return x_r, rgb_comple, rgb_fusion, x_d, depth_comple, depth_fusion

class DRC(nn.Module):
    def __init__(self,in_channel, n_class):
        super(DRC, self).__init__()

        self.rgb_segconv = nn.Conv2d(in_channels=in_channel,out_channels=n_class,kernel_size=1,stride=1,padding=0)
        self.depth_segconv = nn.Conv2d(in_channels=in_channel,out_channels=n_class,kernel_size=1,stride=1,padding=0)


        self.rgb_comple_conv1 = nn.Conv2d(in_channels=n_class,out_channels=n_class,kernel_size=3,stride=1,padding=1)
        self.rgb_comple_bn1 = nn.BatchNorm2d(n_class)
        self.rgb_comple_relu1 = nn.ReLU()
        self.rgb_comple_fusion_conv1 = nn.Conv2d(in_channels=n_class*2,out_channels=n_class,kernel_size=1,stride=1,padding=0)


        self.depth_comple_conv1 = nn.Conv2d(in_channels=n_class,out_channels=n_class,kernel_size=3,stride=1,padding=1)
        self.depth_comple_bn1 = nn.BatchNorm2d(n_class)
        self.depth_comple_relu1 = nn.ReLU()
        self.depth_comple_fusion_conv1 = nn.Conv2d(in_channels=n_class*2,out_channels=n_class,kernel_size=1,stride=1,padding=0)


    def forward(self,fusion,x_r,x_d):

        rgb_missing = self.rgb_segconv(fusion)
        rgb_comple1 = self.rgb_comple_conv1(rgb_missing)
        rgb_comple1 = self.rgb_comple_bn1(rgb_comple1)
        rgb_comple1 = self.rgb_comple_relu1(rgb_comple1)
        rgb_comple = rgb_missing+rgb_comple1
        rgb_fusion = self.rgb_comple_fusion_conv1(torch.cat((x_r,rgb_comple),dim=1))

        depth_missing = self.depth_segconv(fusion)
        depth_comple1 = self.depth_comple_conv1(depth_missing)
        depth_comple1 = self.depth_comple_bn1(depth_comple1)
        depth_comple1 = self.depth_comple_relu1(depth_comple1)
        depth_comple = depth_missing+depth_comple1
        depth_fusion = self.depth_comple_fusion_conv1(torch.cat((x_d,depth_comple),dim=1))

        return rgb_comple,depth_comple, rgb_fusion, depth_fusion 