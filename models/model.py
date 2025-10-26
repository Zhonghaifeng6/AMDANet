import torch.nn as nn
import torch.nn.functional as F
import torch
from engine.logger import get_logger
from .decoder.Seg_Decoder import seg_decoder
from .decoder.Fuse_Decoder import fuse_decoder
from .backbone.Segformer import mit_b4 as backbone


logger = get_logger()
def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group

class AMDANet(nn.Module):
    # def __init__(self, cfg=None, criterion=None, norm_layer=nn.BatchNorm2d, enable_mutual_masking=False):
    def __init__(self, cfg=None, criterion=None, norm_layer=nn.BatchNorm2d):
        super(AMDANet, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # self.enable_mutual_masking = enable_mutual_masking
        self.backbone = backbone(norm_fuse=norm_layer)


        logger.info('Using Segformer-MLP Decoder')
        self.decode_head = seg_decoder(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        self.aux_head = fuse_decoder(in_channels=self.channels)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_backbone)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        logger.info('Initing weights ...')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ori_size = rgb.shape
        ori_inputs = [rgb, modal_x]

        vision_cr = []
        semantic_cr = []
        semantic_r = []
        x_vision_r = []

        x_vision, x_semantic = self.backbone(rgb, modal_x)

        randn_value = torch.randint(1, 3, (1,))
        if randn_value == 1:
            for x_semantic_cr in x_semantic:
                semantic_drop = (torch.rand_like(x_semantic_cr) > 0.15).float()
                drop_cr_sem = x_semantic_cr * semantic_drop
                semantic_cr.append(drop_cr_sem)
            x_vision_r = x_vision
            semantic_r = semantic_cr
        elif randn_value == 2:
            for x_vision_cr in x_vision:
                vision_drop = (torch.rand_like(x_vision_cr) > 0.15).float()
                drop_cr_vis = x_vision_cr * vision_drop
                vision_cr.append(drop_cr_vis)
            x_vision_r = vision_cr
            semantic_r = x_semantic


        out_semantic_r = self.decode_head.forward(semantic_r)
        out_semantic_r = F.interpolate(out_semantic_r, size=ori_size[2:], mode='bilinear', align_corners=False)

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        if self.aux_head:

            Fus_img = self.aux_head.forward(x_vision, ori_inputs)

            Fus_img_r = self.aux_head.forward(x_vision_r, ori_inputs)

            out = [out_semantic,out_semantic_r, Fus_img, Fus_img_r]
            return out

        return out_semantic

    def forward(self, rgb, modal_x, Mask=None, label=None):
        inputs = [rgb, modal_x]
        if self.aux_head:
            outputs = self.encode_decode(rgb, modal_x)
        else:
            outputs = self.encode_decode(rgb, modal_x)

        if label is not None:            
            if self.aux_head:
                loss = self.criterion(inputs, outputs, Mask, label.long())
            else:
                loss = self.criterion(outputs, Mask, label.long())
            return loss
        return outputs