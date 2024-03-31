import torch
import torch.nn as nn
from .layers import CrossFFN, FeedForwardLayer
import numpy as np
import math
from utils.models_utils import *
import torch.nn.functional as F
import copy

class MultiHeadSpatialNet(nn.Module):
    def __init__(self, view_num, n_head, d_model, d_hidden=2048, dropout=0.1):
        super(MultiHeadSpatialNet, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.view_num = view_num

        self.k_proj= nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        # Create a linear layer for each head
        self.score_layers = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_head)])
        self.ffn = FeedForwardLayer(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, spatial_features):
        BVN, N, _ = spatial_features.shape
        k = self.k_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)
        v = self.v_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)        
        # Initialize containers for the multihead outputs
        multihead_features = []
        multihead_score = []
        for i in range(self.n_head):
            k_head = k[:, :, i, :]
            v_head = v[:, :, i, :]
            score = self.score_layers[i](k_head).squeeze(-1)
            score = F.softmax(score, dim=-1).unsqueeze(-1)
            feature = (score * v_head).sum(dim=1)
            multihead_features.append(feature)
            multihead_score.append(score)
        # Concatenate results from all heads
        feature = torch.cat(multihead_features, dim=-1)
        score = torch.cat(multihead_score, dim=-1).mean(dim=-1)        
        features = self.ffn(feature)
        
        return features
    
class FusionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.view_num = config.view_num
        self.n_layer = config.n_layer
        self.late_view_agg =  config.late_view_agg
        self.norm_xy = config.norm_xy
        self.norm_z = config.norm_z
        self.norm_d = config.norm_d
        self.spatial_enc = nn.Sequential(nn.Linear(8, self.d_model),
                                         nn.Dropout(config.spatial_enc.dropout_rate),
                                         nn.LayerNorm(self.d_model))
        layer = CrossFFN(n_head=config.obj_text.n_head, d_model=self.d_model, d_hidden=config.obj_text.d_hidden, dropout=config.obj_text.dropout_rate)
        self.obj_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layer)])
        self.spatial_text = CrossFFN(n_head=config.spatial_text.n_head, d_model=self.d_model, d_hidden=config.spatial_text.d_hidden, dropout=config.spatial_text.dropout_rate)
        layer = MultiHeadSpatialNet(view_num=self.view_num, n_head=config.sp_agg.n_head, d_model=self.d_model, d_hidden=config.sp_agg.d_hidden, dropout=config.sp_agg.dropout_rate)
        self.sp_agg = nn.ModuleList([copy.deepcopy(layer)
                                            for _ in range(config.n_layer)])
    
    @torch.no_grad()
    def get_pairwise_distance(self, x):
        B, N, _ = x.shape
        relative_positions = x[:, None] - x[:, :, None]        
        # Obtain the xy distances
        xy_distances = relative_positions[..., :2].norm(dim=-1, keepdim=True) + 1e-9
        r = xy_distances.squeeze(-1)
        phi = torch.atan2(relative_positions[..., 1], relative_positions[..., 0])  # Azimuth angle
        theta = torch.atan2(r, relative_positions[..., 2])  # Elevation angle
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        relative_positions = torch.cat([relative_positions, xy_distances, sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-1), 
                                    sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)
        # Normalize x-y plane to unit vectors
        if self.norm_xy:
            relative_positions[..., :2] = relative_positions[..., :2] / xy_distances
        # Scale z values so that max(z) - min(z) = 1
        if self.norm_z:
            relative_positions[..., 2] = scale_to_unit_range(relative_positions[..., 2])
        # Scale d values between 0 and 1 for each set of relative positions independently.
        if self.norm_d:
            relative_positions[..., 3] = scale_to_unit_range(relative_positions[..., 3])
        return relative_positions

    def forward(self, objs, text, boxes, device='cuda'):
        """
        objs: (B, N, D)
        text: (B, seq_len, D)
        boxes: (B, N, 7)
        """        
        self.device = device
        B, N, _ = boxes.shape 
        # Data view
        boxes = aug_box(boxes, self.view_num, self.training, self.device)
        boxes = boxes.reshape(B*self.view_num, N, 7)
        xyz = boxes[...,:3]
        
        # Get relative positions and pos features
        relative_positions = self.get_pairwise_distance(xyz).detach()
        spatial_features = self.spatial_enc(relative_positions).reshape(B*self.view_num*N, N, self.d_model)
        # Forward
        text_view_N = batch_expansion(batch_expansion(text, self.view_num), N)
        spatial_features = self.spatial_text(spatial_features, text_view_N)
        
        for i in range(self.n_layer):
            objs = self.obj_layers[i](objs, text)
            objs_view_N = batch_expansion(batch_expansion(objs, self.view_num), N)
            spatial_agg = self.sp_agg[i](spatial_features+objs_view_N)
            spatial_agg = rotation_aggregate(spatial_agg.reshape(B, self.view_num, N, self.d_model))
            objs = objs + spatial_agg
        return objs