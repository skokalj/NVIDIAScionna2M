import copy
import random
import re
import json
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from loss_methods import ClipLoss, SigLipLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SWEEP_CONFIGS = {
    "run_A": {
        "lr": 1e-4,
        "lr_proj": 3e-4,
        "weight_decay": 1e-3,
        "description": "Baseline lr, high WD",
    },
    "run_B": {
        "lr": 5e-5,
        "lr_proj": 1.5e-4,   # keep lr_proj ~3x lr ratio
        "weight_decay": 1e-3,
        "description": "Lower lr, high WD",
    },
    "run_C": {
        "lr": 1e-4,
        "lr_proj": 3e-4,
        "weight_decay": 1e-4,
        "description": "Baseline lr, mid WD",
    },
}



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
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
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
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return F.normalize(x, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, dropout_rate=0.3,
                 num_fc_layers=2, transpose=False):
        super().__init__()
        self.num_layers = num_fc_layers
        self.transpose = transpose
        layers = nn.ModuleList()

        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_fc_layers - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(output_dim, output_dim))

        self.layers = layers
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and self.transpose:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        return x


def clip_loss(zr, xml, logit_scale):
    logit_scale = logit_scale.exp()
    logit_scale = torch.clamp(logit_scale, max=100)
    logits = logit_scale * zr @ xml.T
    labels = torch.arange(zr.size(0), device=zr.device)
    loss = (
        F.cross_entropy(logits, labels, label_smoothing=0.1) +
        F.cross_entropy(logits.T, labels, label_smoothing=0.1)
    ) / 2
    return loss, logits



class SceneClipModelPad(pl.LightningModule):
    def __init__(
        self,
        xml_model_emb_dim=128,
        zr_model_emb_dim=128,
        projected_emb_dim=64,
        dropout_rate=0.3,
        num_fc_layers=1,
        lr=1e-4,
        lr_frac_lm=0,
        lr_proj=3e-4,
        weight_decay=1e-6,
        n_chans=21,
        warmup_epochs=10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_proj = lr_proj
        self.lr_lm = self.lr * lr_frac_lm
        self.weight_decay = weight_decay
        self.n_chans = n_chans
        self.warmup_epochs = warmup_epochs
        self.zr_model_emb_dim = zr_model_emb_dim
        self.xml_model_emb_dim = xml_model_emb_dim

        self.zr_encoder = ZREncoder(zr_model_emb_dim)
        self.xml_encoder = ZXMLEncoder(xml_model_emb_dim)

        self.xml_projection = ProjectionHead(
            input_dim=self.xml_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
        )
        self.zr_projection = ProjectionHead(
            input_dim=zr_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
        )

        init_logit_scale = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        self.features_train = []
        self.features_valid = []

    def forward(self, batch):
        zr_batch, xml_batch = batch
        zr_features = self.zr_encoder(zr_batch)
        xml_features = self.xml_encoder(xml_batch)
        zr_features_proj = self.zr_projection(zr_features)
        xml_features_proj = self.xml_projection(xml_features)
        return zr_features, zr_features_proj, xml_features, xml_features_proj

    def training_step(self, batch, batch_idx):
        _, zr_features_proj, _, xml_features_proj = self.forward(batch)
        self.features_train.append(zr_features_proj)
        loss, _ = clip_loss(zr_features_proj, xml_features_proj, self.logit_scale)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, zr_features_proj, _, xml_features_proj = self.forward(batch)
        self.features_valid.append(zr_features_proj)
        loss, _ = clip_loss(zr_features_proj, xml_features_proj, self.logit_scale)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        features_valid = torch.cat(self.features_valid).cpu()
        self.features_valid.clear()

    def configure_optimizers(self):
        params = list(self.named_parameters())

        grouped_parameters = [
            {"params": [p for n, p in params if "encoder" in n],    "lr": self.lr},
            {"params": [p for n, p in params if "projection" in n], "lr": self.lr_proj},
            {"params": [p for n, p in params if "logit_scale" in n],"lr": 1e-5},
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # Linear warmup then cosine annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]



def compute_norm_stats(X):
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
    return indices[n_val:], indices[:n_val]


def build_dataloaders(batch_size=128):
    xmldata = np.load("/data/hafeez/z_xml.npy").astype(np.float32)
    zrdata  = np.load("/data/hafeez/zrs1.npy").astype(np.float32)

    SEQ_LEN, PAD_TO = 107, 112
    assert zrdata.shape[-1] == SEQ_LEN
    z_pad = np.pad(zrdata, ((0, 0), (0, PAD_TO - SEQ_LEN)), constant_values=0)

    N = len(zrdata)
    train_idx, val_idx = train_val_split_indices(N)

    train_A = [z_pad[i]   for i in train_idx]
    train_B = [xmldata[i] for i in train_idx]
    val_A   = [z_pad[i]   for i in val_idx]
    val_B   = [xmldata[i] for i in val_idx]

    # FIX: compute stats on TRAIN data only, apply same transform to val
    AT_max, AT_mean, AT_std = compute_norm_stats(train_A)
    BT_max, BT_mean, BT_std = compute_norm_stats(train_B)

    transform_A = NormalizeTransform(AT_max, AT_mean, AT_std)
    transform_B = NormalizeTransform(BT_max, BT_mean, BT_std)

    train_dataset = PairedContrastiveDataset(train_A, train_B, transform_A, transform_B)
    val_dataset   = PairedContrastiveDataset(val_A,   val_B,   transform_A, transform_B)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    valloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, valloader



def run_sweep(run_name, cfg, trainloader, valloader, num_epochs=300, num_gpus=8):
    print(f"\n{'='*60}")
    print(f"  Starting {run_name}: {cfg['description']}")
    print(f"  lr={cfg['lr']}  lr_proj={cfg['lr_proj']}  weight_decay={cfg['weight_decay']}")
    print(f"{'='*60}\n")

    model = SceneClipModelPad(
        n_chans=1,
        lr=cfg["lr"],
        lr_proj=cfg["lr_proj"],
        lr_frac_lm=0,
        weight_decay=cfg["weight_decay"],
        num_fc_layers=3,
        projected_emb_dim=128,
        xml_model_emb_dim=128,
        zr_model_emb_dim=128,
        warmup_epochs=10,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    logger = pl.loggers.CSVLogger("logs", name=run_name)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=num_gpus,
        strategy="ddp",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

    print(f"\n  {run_name} done. Best val_loss: {checkpoint_cb.best_model_score:.4f}")
    print(f"  Best ckpt: {checkpoint_cb.best_model_path}\n")

    return {
        "run": run_name,
        "description": cfg["description"],
        "lr": cfg["lr"],
        "lr_proj": cfg["lr_proj"],
        "weight_decay": cfg["weight_decay"],
        "best_val_loss": float(checkpoint_cb.best_model_score),
        "best_ckpt": checkpoint_cb.best_model_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   nargs="+", default=["run_A", "run_B", "run_C"],
                        help="Which runs to execute (e.g. --runs run_A run_C)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gpus",   type=int, default=8)
    parser.add_argument("--batch",  type=int, default=128)
    args = parser.parse_args()

    trainloader, valloader = build_dataloaders(batch_size=args.batch)

    results = []
    for run_name in args.runs:
        if run_name not in SWEEP_CONFIGS:
            print(f"Unknown run '{run_name}', skipping. Valid: {list(SWEEP_CONFIGS)}")
            continue
        result = run_sweep(
            run_name,
            SWEEP_CONFIGS[run_name],
            trainloader,
            valloader,
            num_epochs=args.epochs,
            num_gpus=args.gpus,
        )
        results.append(result)

    # ── Summary table ──
    print("\n" + "="*60)
    print("  SWEEP SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)[["run", "description", "lr", "weight_decay", "best_val_loss"]]
    print(df.to_string(index=False))
    df.to_csv("sweep_results.csv", index=False)
    print("\nResults saved to sweep_results.csv")