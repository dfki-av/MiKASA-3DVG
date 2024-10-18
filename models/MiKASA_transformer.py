import torch
from torch import nn
from .point_net_pp import PointNetPP
from .fusion_net import FusionNet
from transformers import BertModel, BertConfig
from utils.models_utils import *

class MiKASA_transformer(nn.Module):
    def __init__(self,
                 config,
                 n_obj_classes,
                 ignore_index):

        super().__init__()
        self.view_number = config.view_number
        self.rotate_number = config.rotate_number
        self.lang_cls_alpha = config.lang_cls_alpha
        self.obj_cls_alpha = config.obj_cls_alpha
        self.d_model = config.d_model
        self.n_obj_classes = n_obj_classes
        self.fl_weight = config.fl_weight
        self.cl_weight = config.cl_weight
        self.loss_l = config.loss_l
        self.loss_o = config.loss_o
        self.loss_po = config.loss_po
             
        # Object
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, config.d_model, config.d_model]]])
        self.obj_feature_mapping = create_mapping(config.d_model, config.d_model, config.dropout_rate)
        self.box_feature_mapping = create_mapping(7, config.d_model, config.dropout_rate)
        self.post_obj_enc = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=config.d_model, 
                                                nhead=8, dim_feedforward=config.d_hidden, activation="gelu"), num_layers=config.post_obj_layers)
        
        # Lauguage
        self.language_encoder = BertModel.from_pretrained(config.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:config.text_encoder.encoder_layer_num]
        
        # fusion_net
        self.fusion_net = FusionNet(config.fusion) 
        
        # Classifier heads
        self.object_clf = create_classifier(config.d_model, self.n_obj_classes, config.dropout_rate)
        self.post_object_clf = create_classifier(config.d_model, self.n_obj_classes, config.dropout_rate)
        self.language_clf = create_classifier(config.d_model, self.n_obj_classes, config.dropout_rate)
        self.fusion_clf = create_classifier(config.d_model, 1, config.dropout_rate)        
        
        # Loss
        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.class_logits_post_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        post_obj_clf_loss = self.class_logits_post_loss(POST_CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
        total_loss = referential_loss + self.loss_o * obj_clf_loss + self.loss_l * lang_clf_loss + self.loss_po*post_obj_clf_loss
        return total_loss
        
    def forward(self, batch: dict, epoch=None):
        self.device = self.obj_feature_mapping[0].weight.device
        ## rotation augmentation and multi_view generation
        obj_points = aug_input(batch['objects'], (0.5, 1.5), 0.02, self.rotate_number, self.training, self.device)
        B,R,N,P = obj_points.shape[:4]
        obj_points = obj_points.reshape(B*R,N,P,6)
        boxes = aug_box(batch['box_info'], self.rotate_number, self.training, self.device)  
        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)      
        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)    
        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]
        # <LOSS>: obj_cls        
        CLASS_LOGITS = rotation_aggregate(self.object_clf(obj_feats).reshape(B,R,N,self.n_obj_classes))
        # <LOSS>: post_obj_cls
        obj_feats = obj_feats + self.box_feature_mapping(boxes).reshape(B*R, N, self.d_model)
        obj_feats = self.post_obj_enc(obj_feats.transpose(0,1)).transpose(0,1)        
        POST_CLASS_LOGITS = rotation_aggregate(self.post_object_clf(torch.clone(obj_feats)).reshape(B,R,N,self.n_obj_classes))
        # <LOSS>: lang_cls
        lang_features = lang_infos[:,0]
        LANG_LOGITS = self.language_clf(torch.clone(lang_features))   
        max_indices_lang_logits = torch.argmax(LANG_LOGITS, dim=1)
        expanded_indices = max_indices_lang_logits.unsqueeze(1).expand(B, N)
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
        cate_logits = POST_CLASS_LOGITS[batch_indices, torch.arange(N), expanded_indices]        
        cate_logits = norm_output_scores(cate_logits)
        ## spatial_encoding
        fusion_info = self.fusion_net(objs=rotation_aggregate(torch.clone(obj_feats).reshape(B,R,N,self.d_model)) , 
                                    text=lang_infos, 
                                    boxes=batch['box_info'])
        # <LOSS>: ref_cls
        fusion_logits = norm_output_scores(self.fusion_clf(fusion_info).squeeze(-1))
        
        LOGITS = self.fl_weight*fusion_logits + self.cl_weight*cate_logits
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS)
        return LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS
