import copy
import random
import re
import json
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
#from braindecode.models import Deep4Net
#from braindecode.models.util import to_dense_prediction_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

#import configs.preprocess_config as preprocess_config
from loss_methods import ClipLoss, SigLipLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ZXMLEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            ResBlock1D(128),
            ResBlock1D(128),
            #ResBlock1D(128),
            #ResBlock1D(128),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B,128,1]
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: [B, 576]
        x = x.unsqueeze(1)          # [B,1,576]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return F.normalize(x, dim=-1)

class ZREncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
                

        self.blocks = nn.Sequential(
            ResBlock1D(128),
            ResBlock1D(128),
            #ResBlock1D(128),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: [B, 107]
        x = x.unsqueeze(1)          # [B,1,107]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return F.normalize(x, dim=-1)

class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim=128,
        output_dim=128,
        dropout_rate=0.3,
        num_fc_layers=2,
        transpose=False,
    ):
        super().__init__()
        """
        self.projection_layer = nn.Linear(input_dim, output_dim)
        self.fc_layers = nn.ModuleList()

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(output_dim, output_dim))
        self.layer_norm = nn.LayerNorm(
            output_dim
        )  # TODO : what is the benefit of layer norm here?
        self.transpose = transpose

    def forward(self, x):
        x_proj = self.projection_layer(x)
        for layer in self.fc_layers:
            x_proj_fc = self.dropout(self.gelu(layer(x_proj)))
            x_proj = x_proj + x_proj_fc
            x_proj = self.layer_norm(x_proj)

        if self.transpose:
            x_proj = x_proj.transpose(1, 2)  # [B,Enc_size,N_pred]
        return x_proj
        """
        super(ProjectionHead, self).__init__()
        self.num_layers = num_fc_layers
        self.transpose = transpose
        layers = nn.ModuleList()

        # Input projection layer
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_fc_layers - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output projection layer
        layers.append(nn.Linear(output_dim, output_dim))

        self.layers = layers
        self.projection = nn.Sequential(*layers)
    def forward(self, x):
        for layer in self.layers:
            # check if layer is batchnorm
            if isinstance(layer, nn.BatchNorm1d) and self.transpose:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
            
        return x
        #return self.projection(x)
def clip_loss(zr, xml, logit_scale):
    # zr, xml: [B, D]
    logit_scale = logit_scale.exp()
    logit_scale = torch.clamp(logit_scale, max=100)

    logits = logit_scale * zr @ xml.T
    labels = torch.arange(zr.size(0), device=zr.device)

    loss = (
        F.cross_entropy(logits, labels,label_smoothing=0.1) +
        F.cross_entropy(logits.T, labels, label_smoothing=0.1)
    ) / 2

    return loss, logits

class SceneClipModelPad(pl.LightningModule):
    def __init__(
        self,
        xml_model_emb_dim=128,
        zr_model_emb_dim=128,
        projected_emb_dim=64,
        zr_model_pretrained=False,
        zr_model_trainable=True,
        xml_model_pretrained=False,
        xml_model_trainable=True,
        dropout_rate=0.3,
        num_fc_layers=1,
        lr=1e-4,
        lr_frac_lm=0,
        lr_proj=3e-4,
        weight_decay=1e-6,
        n_chans=21,
        contrastive_loss_temperature=0.07, #1
        contrastive_loss_func="clip",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_proj = lr_proj
        self.lr_lm = self.lr * lr_frac_lm
        self.weight_decay = weight_decay
        self.n_chans = n_chans
        self.contrastive_loss_temperature = contrastive_loss_temperature
        self.contrastive_loss_func = contrastive_loss_func
        self.zr_model_emb_dim = zr_model_emb_dim
        self.xml_model_emb_dim = xml_model_emb_dim
        #print(self.text_encoder_emb_dim)

        self.zr_encoder = ZREncoder(
            zr_model_emb_dim
        )
        self.xml_encoder = ZXMLEncoder(
            xml_model_emb_dim
        )

        self.xml_projection = ProjectionHead(
            input_dim=self.xml_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
            transpose = False,
        )

        self.zr_projection = ProjectionHead(
            input_dim=zr_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
            transpose=False,
        )
        init_logit_scale = np.log(1 / 0.07)
        '''
        if contrastive_loss_func == "clip":
            self.loss_fn = ClipLoss()
            init_logit_scale = np.log(1 / 0.07)
            init_logit_bias = 0
        elif contrastive_loss_func == "siglip":
            self.loss_fn = SigLipLoss()
            init_logit_scale = np.log(10)
            init_logit_bias = -10
            '''

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        #self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

        # save features and labels for classification
        self.features_train = []
        self.labels_train = []

        self.features_valid = []
        self.labels_valid = []

        #self.report_list = []

    def forward(self, batch):
        zr_batch, xml_batch = batch
        #self.report_list.extend(list(string_batch))
        zr_features = self.zr_encoder(zr_batch)
        xml_features = self.xml_encoder(xml_batch)

        # print("PROJECTING ZR FEATURES")
        zr_features_proj = self.zr_projection(zr_features)
        #zr_features_proj = torch.mean(zr_features_proj, dim=1) # average over the time dimension
        # print("PROJECTING XML FEATURES")
        xml_features_proj = self.xml_projection(xml_features)
        #xml_features_proj = torch.mean(xml_features_proj, dim=1) # average over the time dimension


        return (
            zr_features,
            zr_features_proj,
            xml_features,
            xml_features_proj,
        )

    def training_step(self, batch, batch_idx):
        (
            zr_features,
            zr_features_proj,
            xml_features,
            xml_features_proj,
        ) = self.forward(batch)
        self.features_train.append(zr_features_proj)
        #self.labels_train.append(labels)

        # loss = self.loss_calculation(eeg_features_proj, text_features_proj,self.contrastive_loss_temperature)
        #loss, logits_per_image = self.loss_fn(
        #    zr_features_proj, xml_features_proj, self.logit_scale, self.logit_bias
        #)
        loss, logits = clip_loss(
            zr_features_proj,
            xml_features_proj,
            self.logit_scale
        )

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            zr_features,
            zr_features_proj,
            xml_features,
            xml_features_proj,
        ) = self.forward(batch)
        self.features_valid.append(zr_features_proj)

        #loss, logits_per_image = self.loss_fn(
        #    zr_features_proj, xml_features_proj, self.logit_scale, self.logit_bias
        #)
        loss, logits=clip_loss(
            zr_features_proj,
            xml_features_proj,
            self.logit_scale
        )
        self.log("val_loss", loss, prog_bar=True)

        #print("logits")
        #print(logits.shape)

        return loss

    def test_step(self, batch, batch_idx):
        (
            zr_features,
            zr_features_proj,
            xml_features,
            xml_features_proj,
        ) = self.forward(batch)
        self.features_valid.append(zr_features_proj)

        #loss, logits_per_image = self.loss_fn(
        #    zr_features_proj, xml_features_proj, self.logit_scale, self.logit_bias
        #)
        self.log("test_loss", loss, prog_bar=True)

        #print("logits_per_image")
        #print(logits_per_image.shape)

        return loss        

    def on_validation_epoch_end(self):
        # report_list = list(set(self.report_list))
        # with open('parrot.pkl', 'wb') as f:
        #   pickle.dump(report_list, f)

        features_valid = torch.cat(self.features_valid).cpu()

        return None

    def configure_optimizers(self):
        params = list(self.named_parameters())
        print("params")
        print([n for n, p in params])

        def is_encoder(n):
            return "encoder" in n
        def is_projection(n):
            return "projection" in n
        def is_logit(n):
            return "logit_scale" in n

        grouped_parameters = [
            {"params": [p for n, p in params if is_encoder(n)], "lr": self.lr},
            {"params": [p for n, p in params if is_projection(n)], "lr": self.lr_proj},
            {"params": [p for n, p in params if is_logit(n)], "lr": 1e-5},            
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.98), eps=1e-8  
        )

        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs - 1
        )
        return [optimizer], [scheduler]


def on_save_checkpoint(checkpoint):
    for key in list(checkpoint["state_dict"].keys()):
        if "text_encoder" in key:
            print("deleting ", key)
            del checkpoint["state_dict"][key]

#xmldata = np.load("/data/silvija/data/z_xml.npy").astype(np.float32)
xmldata = np.load("/data/hafeez/z_xml.npy").astype(np.float32)
zrdata = np.load("/data/hafeez/zrs1.npy").astype(np.float32)
SEQ_LEN=107
PAD_TO=112
assert zrdata.shape[-1] == SEQ_LEN
pad_right = PAD_TO - SEQ_LEN
z_pad = np.pad(zrdata, ((0, 0), (0, pad_right)), constant_values = 0)  
def compute_norm_stats(X):
    #X = np.stack([x.squeeze(0) for x in list_X], axis=0)  # [N, D]
    Xn = X / np.max(X)
    m = np.mean(Xn, axis=0)
    s = np.std(Xn, axis=0) + 1e-8
    return np.max(X), m, s

class NormalizeTransform:
    def __init__(self, max_val, mean, std):
        self.max_val = max_val
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, x):
        x = x / self.max_val
        return (x - self.mean) / self.std

from torch.utils.data import Dataset
class PairedContrastiveDataset(Dataset):
    def __init__(self, list_A, list_B, transform_A=None, transform_B=None):
        assert len(list_A) == len(list_B)
        self.list_A = list_A
        self.list_B = list_B
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __len__(self):
        return len(self.list_A)

    def __getitem__(self, idx):
        A = torch.tensor(self.list_A[idx], dtype=torch.float32).squeeze(0)
        B = torch.tensor(self.list_B[idx], dtype=torch.float32).squeeze(0)

        if self.transform_A:
            A = self.transform_A(A)
        if self.transform_B:
            B = self.transform_B(B)

        return A, B

def train_val_split_indices(n, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return train_idx, val_idx
N = len(zrdata)
train_idx, val_idx = train_val_split_indices(N, val_ratio=0.2)
train_A = [z_pad[i] for i in train_idx]
train_B = [xmldata[i] for i in train_idx]

val_A = [z_pad[i] for i in val_idx]
val_B = [xmldata[i] for i in val_idx]

AT_max, AT_mean, AT_std = compute_norm_stats(train_A)
BT_max, BT_mean, BT_std = compute_norm_stats(train_B)
AV_max, AV_mean, AV_std = compute_norm_stats(val_A)
BV_max, BV_mean, BV_std = compute_norm_stats(val_B)

transform_AT = NormalizeTransform(AT_max, AT_mean, AT_std)
transform_BT = NormalizeTransform(BT_max, BT_mean, BT_std)
transform_AV = NormalizeTransform(AV_max, AV_mean, AV_std)
transform_BV = NormalizeTransform(BV_max, BV_mean, BV_std)
from torch.utils.data import random_split
lr=5e-3
lr=1e-4
lr_proj=3e-4
weight_decay=1e-6
n_chans=1
num_fc_layers=3
lr_frac_lm=0
projected_emb_dim=128
xml_model_emb_dim=zr_model_emb_dim=128
zr_model_emb_dim
num_epochs=400

train_dataset = PairedContrastiveDataset(train_A, train_B, transform_A=transform_AT, transform_B=transform_BT,)
val_dataset   = PairedContrastiveDataset(val_A, val_B,
                                        transform_A=transform_AV, transform_B=transform_BV,)

trainloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=128,
                    shuffle=True)

valloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=128,
                    shuffle=True)
trainer = pl.Trainer(max_epochs=num_epochs,
                     accelerator="auto", devices=8)

trainer.fit(SceneClipModelPad(
            n_chans=n_chans,
            lr=lr,
            lr_proj = lr_proj,
            lr_frac_lm=lr_frac_lm,
            weight_decay=weight_decay,
            num_fc_layer=num_fc_layers,
            projected_emb_dim=projected_emb_dim,
            xml_model_emb_dim=xml_model_emb_dim,
            zr_model_emb_dim=zr_model_emb_dim,
        ), train_dataloaders=trainloader,
                    val_dataloaders=valloader)