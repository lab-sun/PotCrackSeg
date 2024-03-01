import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from util.init_func import init_weight
from util.load_utils import load_pretrain
from functools import partial
from config import config



class EncoderDecoder(nn.Module):
    def __init__(self,cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), encoder_name='mit_b2', decoder_name='Decoder_RGB', n_class=3, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

        # import backbone and decoder
        if encoder_name == 'PotCrackSeg-5B':
            #logger.info('Using backbone: Segformer-B5')
            from model.mysegformer.encoders.dual_segformer import mit_b5 as backbone
            print("chose mit_b5")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'PotCrackSeg-4B':
            #logger.info('Using backbone: Segformer-B4')
            from model.mysegformer.encoders.dual_segformer import mit_b4 as backbone
            print("chose mit_b4")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'PotCrackSeg-3B':
            #logger.info('Using backbone: Segformer-B4')
            from model.mysegformer.encoders.dual_segformer import mit_b3 as backbone
            print("chose mit_b3")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'PotCrackSeg-2B':
            #logger.info('Using backbone: Segformer-B2')
            from model.mysegformer.encoders.dual_segformer import mit_b2 as backbone
            print("chose mit_b2")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'PotCrackSeg-1B':
            #logger.info('Using backbone: Segformer-B1')
            from model.mysegformer.encoders.dual_segformer import mit_b1 as backbone
            print("chose mit_b1")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'PotCrackSeg-0B':
            #logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from model.mysegformer.encoders.dual_segformer import mit_b0 as backbone
            print("chose mit_b0")
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            #logger.info('Using backbone: Segformer-B2')
            from encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None


        if decoder_name == 'Decoder_RGB':
            from model.mysegformer.decoders.Decoder_RGB import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)
        else: 
            from model.mysegformer.decoders.Decoder_Depth import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)



        self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        rgb,depth = self.backbone(rgb, modal_x)
        out_x = self.decode_head.forward(rgb,depth)
        out_x = F.interpolate(out_x, size=orisize[2:], mode='bilinear', align_corners=False)


        return out_x

    def forward(self, input):

        rgb = input[:,:3]
        modal_x = input[:,3:]
        modal_x = torch.cat((modal_x,modal_x,modal_x),dim=1)

        out_x = self.encode_decode(rgb, modal_x)

        return out_x


def unit_test():

    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 288, 512).cuda(1)
    thermal = torch.randn(num_minibatch, 1, 288, 512).cuda(1)
    images = torch.cat((rgb,thermal),dim=1)
    rtf_net = EncoderDecoder(cfg = config, encoder_name='mit_b4', decoder_name='MLPDecoderonlyDepth', n_class=3).cuda(1)
    #input = torch.cat((rgb, thermal), dim=1)
    #out_seg_4,out_seg_3,out_seg_2,out_seg_1,out_x,out_d,final = rtf_net(images)

    rtf_net.eval()

    final = rtf_net(images)

    #out_seg_4,out_seg_3,out_seg_2,out_seg_1,out_x,out_d,final = final

    # for i in out_seg_1:
    #     print(i.shape)


    print(final.shape)

if __name__ == '__main__':
    unit_test()
