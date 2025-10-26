import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
import torch.nn.functional as F


#######################################################################################
#######   ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Semantic Consistency Inference ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓   #######
#######################################################################################
class SemanticConsistencyInference(nn.Module):
    def __init__(self, input_dim, lambda_init=0.5):
        super(SemanticConsistencyInference, self).__init__()

        # MLP for generating Msc
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # Concatenate Fin and Fvi
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

        # Learnable parameter λ
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))  # Lambda is a learnable scalar
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fin, Fvi, t = 0.4):

        cos_sim = F.cosine_similarity(Fin, Fvi, dim=-1)  # Shape [B, L]
        Sm = 0.1 * cos_sim.mean(dim=-1)  # Mean similarity for each sample in the batch [B]

        if torch.any(Sm < t):  # If there are any elements where Sm < t, apply difference elimination

            # Concatenate Fin and Fvi to form a common feature set
            concatenated = torch.cat((Fin, Fvi), dim=-1)  # Shape [B, L, 2*C]

            Mmid = self.mlp(concatenated).squeeze(-1)
            Msc = self.sigmoid(Mmid)  # Shape [B, L]

            Pin = Fin * (1 - Msc.unsqueeze(-1)) + Fvi * Msc.unsqueeze(-1)
            Pvi = Fvi * (1 - Msc.unsqueeze(-1)) + Fin * Msc.unsqueeze(-1)
            Kvi = Fvi - Pvi
            Kin = Fin - Pin

            aFvi = Fvi - self.lambda_param * Kvi
            aFin = Fin - self.lambda_param * Kin
        else:
            aFvi, aFin = Fvi, Fin

        return aFvi, aFin


#######################################################################################
####   ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Feature Discrepancy Alignment Module ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓   ####
#######################################################################################

#######  Eliminating discrepancies from local  #######

class From_Channel(nn.Module):

    def __init__(self, dim, reduction=4):
        super(From_Channel, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2) #B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)
        
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        
        Stat_out = self.sigmoid(avg_se+max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights



class From_Spatial(nn.Module):

    def __init__(self, kernel_size=1, reduction=4):
        super(From_Spatial, self).__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size), 
                    nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True)
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True)
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x_cat = torch.cat((x1_mean_out, x1_max_out,x2_mean_out,x2_max_out), dim=1)
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class Fuse_s_c(nn.Module):

    def __init__(self, dim):
        super(Fuse_s_c, self).__init__()
        self.dim = dim
        self.cha = From_Channel(self.dim)
        self.sap = From_Spatial(reduction=4)
    def forward(self, x1, x2):
        f_cha = self.cha(x1, x2)
        f_sap = self.sap(x1, x2)
        mixatt_out = f_cha.mul(f_sap)
        return mixatt_out


class Local_Eliminating(nn.Module):

    def __init__(self, dim, reduction=4):
        super(Local_Eliminating, self).__init__()
        self.fuse_sc = Fuse_s_c(dim)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(
                    nn.Linear(dim * 2, dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim * 2 // reduction, dim),
                    nn.Sigmoid())
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        mid_feature = self.gate(torch.cat((x1_flat, x2_flat), dim=2))
        mid_feature = mid_feature.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous()
        fusion = self.fuse_sc(x1,x2)
        channel_feature = mid_feature * fusion[0]
        spatial_feature = mid_feature * fusion[1]
        out_x1 = x1 + channel_feature * x2
        out_x2 = x2 + spatial_feature * x1
        out_1 = self.sigmoid(out_x1 * channel_feature - out_x1) * out_x1 + out_x1 * channel_feature
        out_2 = self.sigmoid(out_x2 * spatial_feature - out_x2) * out_x2 + out_x2 * spatial_feature
        return out_1, out_2



#######  Eliminating discrepancies from global scope  #######

class Salient_Enhancement(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):

        super(Salient_Enhancement, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten+ x) 
        x_out = self.proj_drop(x_out)
        return x_out


class Cross_Modal_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Cross_Modal_Attention, self).__init__()
        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)                    
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x1, x2, H, W):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x2_ = x2.permute(0, 2, 1).reshape(B2, C2, H, W) 
            x2_ = self.sr(x2_).reshape(B2, C2, -1).permute(0, 2, 1) 
            x2_ = self.norm(x2_)
            kv2 = self.kv2(x2_).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten+x1)
        x_out = self.proj_drop(x_out)
        return x_out


class Global_Eliminating(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Global_Eliminating, self).__init__()
        self.SE_x1 = Salient_Enhancement(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.SE_x2 =  Salient_Enhancement(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CM_x1toX2 = Cross_Modal_Attention(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CM_x2toX1 = Cross_Modal_Attention(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)

        x1_self_enhance = self.SE_x1(x1_flat, H1, W1)
        x2_self_enhance = self.SE_x2(x2_flat, H1, W1)
        x1_cross_enhance = self.CM_x1toX2(x1_self_enhance, x2_self_enhance, H1, W1)
        x2_cross_enhance = self.CM_x2toX1(x2_self_enhance, x1_cross_enhance, H1, W1)
        Fuse = self.proj(x2_cross_enhance)
        Fuse_out = Fuse.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()

        return Fuse_out