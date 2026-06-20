import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import math

# 1.reward从mean换成min 2.reward增加利用率

class RAN_O(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        
        super(RAN_O, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.val_w = nn.Linear(self.in_dim, self.out_dim, bias=False)
        
        self.attn_src = nn.Parameter(torch.empty(size=(self.out_dim, 1)))
        self.attn_dst = nn.Parameter(torch.empty(size=(self.out_dim, 1)))
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)

        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )
 
        if in_dim != out_dim:
            self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_fc = None
        
        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.nega_slope)
        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.val_w.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dst, gain=gain)
        
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight)
    
    def forward(self, opes_mask_adj, opes_mask_job, batch_idxes, feats):

        feat_ope = self.feat_drop(feats)
        h_ope = self.ope_w(feat_ope)
        v_ope = self.val_w(feat_ope)        

        attn_src = torch.matmul(h_ope, self.attn_src).squeeze(-1)
        attn_dst = torch.matmul(h_ope, self.attn_dst).squeeze(-1)
        e = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)
        e = self.leaky_relu(e) 
        
        mask_adj = (opes_mask_adj[batch_idxes] == 1)
        e_adj = e.masked_fill(~mask_adj, float('-9e10'))
        alpha_adj = F.softmax(e_adj, dim=-1)
        alpha_adj = self.attn_drop(alpha_adj)
        h_adj = torch.matmul(alpha_adj, v_ope)
        
        mask_job = (opes_mask_job[batch_idxes] == 1)
        e_job = e.masked_fill(~mask_job, float('-9e10'))
        alpha_job = F.softmax(e_job, dim=-1)
        alpha_job = self.attn_drop(alpha_job)
        h_job = torch.matmul(alpha_job, v_ope)

        h_fus = torch.cat((h_adj, h_job), dim=-1)
        h_fus = self.fusion(h_fus)

        if self.res_fc is not None:
             h_res = self.res_fc(feat_ope)
        else:
             h_res = feat_ope

        return h_fus + h_res


class RAN_M(nn.Module):
    
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        
        super(RAN_M, self).__init__()
        self.ope_dim = in_dim[0]
        self.mac_dim = in_dim[1]
        self.out_dim = out_dim
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.ope_dim, self.out_dim, bias=False)
        self.mac_w = nn.Linear(self.mac_dim, self.out_dim, bias=False)
        
        self.ope_alpha = nn.Parameter(torch.empty(size=(self.out_dim, 1)))
        self.mac_alpha = nn.Parameter(torch.empty(size=(self.out_dim, 1)))
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)
        self.activate = torch.tanh
 
        if in_dim[1] != out_dim:
            self.res_fc = nn.Linear(in_dim[1], out_dim, bias=False)
        else:
            self.res_fc = None

        self.norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.nega_slope)
        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.mac_w.weight, gain=gain)

        nn.init.xavier_normal_(self.ope_alpha, gain=gain)
        nn.init.xavier_normal_(self.mac_alpha, gain=gain)

        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight) 
        
    def forward(self, ope_ma_adj, batch_idxes, feats):

        feat_ope = self.feat_drop(feats[0])
        feat_mac = self.feat_drop(feats[1])
        feat_edg = self.feat_drop(feats[2])
            
        h_ope = self.ope_w(feat_ope)
        h_mac = self.mac_w(feat_mac)
        
        # attention coefficients
        attn_ope = torch.matmul(h_ope, self.ope_alpha).squeeze(-1)
        attn_mac = torch.matmul(h_mac, self.mac_alpha).squeeze(-1)
        
        attn_ope = attn_ope.unsqueeze(-1) + attn_mac.unsqueeze(-2)
        e_ijk = self.leaky_relu(attn_ope)

        mask_ijk = (ope_ma_adj[batch_idxes]==1)
        e_ijk = e_ijk.masked_fill(~mask_ijk, float('-9e10'))
        alpha_ijk = F.softmax(e_ijk, dim=-2)
        alpha_ijk_T = alpha_ijk.transpose(1, 2)
        h_ope = torch.matmul(alpha_ijk_T, h_ope)

        if self.res_fc is not None:
             h_res = self.res_fc(feat_mac)
        else:
             h_res = feat_mac
           
        return h_ope + h_res
