import torch
from torch import nn
from layers import SparseNGCNLayer, DenseNGCNLayer
from torch.nn import functional as F


class MixHopNetwork(torch.nn.Module):

    def __init__(self, args, feature_number, class_number, device):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.dropout = self.args.dropout
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        self.device = device  
        self.transformer_encoder = encoders(self.args.encoder_layers, self.args.num_heads, self.args.d_model,
                                            self.args.d_ff)

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):

        self.upper_layers = [
            SparseNGCNLayer(self.feature_number, self.args.layers_1[i - 1], i, self.args.dropout, self.args.device) for
            i
            in range(1, self.order_1 + 1)]
        self.upper_layers = nn.ModuleList(self.upper_layers)

        self.bottom_layers = [
            DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i - 1], i, self.args.dropout,
                           self.args.device) for i in
            range(1, self.order_2 + 1)]
        self.bottom_layers = nn.ModuleList(self.bottom_layers)

        self.bilinear = nn.Bilinear(self.abstract_feature_number_2, self.abstract_feature_number_2, self.args.hidden1)

        self.decoder_disease_lncrna = nn.Sequential(nn.Linear(self.args.hidden1, self.args.hidden2),
                                     nn.ELU(),
                                     nn.Linear(self.args.hidden2, 1)
                                     )
        self.decoder_lncrna_mirna = nn.Sequential(nn.Linear(self.args.hidden1, self.args.hidden2),
                                     nn.ELU(),
                                     nn.Linear(self.args.hidden2, 1)
                                     )
        self.decoder_mirna_disease = nn.Sequential(nn.Linear(self.args.hidden1, self.args.hidden2),
                                     nn.ELU(),
                                     nn.Linear(self.args.hidden2, 1)
                                     )

    def embed(self, normalized_adjacency_matrix, features):

        indices = features['indices'].to(self.device)
        values = features['values'].to(self.device)
        size = features['dimensions']
        features_tensor = torch.sparse_coo_tensor(indices, values, size).to_dense().to(self.device)
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features_tensor) for i in range(self.order_1)], dim=1)
        abstract_features_1 = F.dropout(abstract_features_1, self.dropout, training=self.training)

        abstract_features_2 = torch.cat(
            [self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],
            dim=1)
        feat = F.dropout(abstract_features_2, self.dropout, training=self.training)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)  

        feat = self.transformer_encoder(feat)
        
        return feat

    def forward(self, normalized_adjacency_matrix, features, idx):
        latent_features = self.embed(normalized_adjacency_matrix, features)
        feat_p1 = latent_features[:, idx[0], :]
        feat_p2 = latent_features[:, idx[1], :]
        feat = F.elu(self.bilinear(feat_p1, feat_p2))
        feat = F.dropout(feat, self.dropout, training=self.training)
        predictions_DLI = self.decoder_disease_lncrna(feat)
        predictions_DMI = self.decoder_mirna_disease(feat)
        predictions_MLI = self.decoder_lncrna_mirna(feat)
        return predictions_DLI, predictions_DMI, predictions_MLI, latent_features

class Attention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(0)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v).transpose(1, 2).flatten(2)
        attn = self.proj(attn)
        attn = self.dropout(attn)
        return attn

class encoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, encoder_dropout=0.05):
        super(encoderLayer, self).__init__()
        d_ff = int(d_ff * d_model)
        self.attention_layer = Attention(num_heads, d_model)

        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(encoder_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        new_x = self.attention_layer(x)
        out1 = self.norm1(x + self.dropout(new_x))
        out2 = self.norm2(out1 + self.dropout(self.feedForward(out1)))
        return out2

class encoders(nn.Module):
    def __init__(self, encoder_layers, num_heads, d_model, d_ff):
        super(encoders, self).__init__()
        self.encoder = nn.ModuleList([encoderLayer(num_heads, d_model, d_ff) for _ in range(encoder_layers)])

    def forward(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        return x
