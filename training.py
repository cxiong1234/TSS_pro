import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import FoldingDiff
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
from pathlib import Path
import util
import random
import math
import numpy as np
import os
import sys
import wandb

# Remove AmberTools from path
sys.path = [path for path in sys.path if 'AmberTools' not in path]

class FoldingDiffDataset(Dataset):
    def __init__(self, meta, tensor_file, T, mu_ref=None, s=8e-3):
        self.meta = meta
        self.records = meta.to_records()
        self.T = T

        if Path(tensor_file).exists():
            print(f"Loading conditioned data from {tensor_file}")
            self.data_cache = torch.load(tensor_file)
        else:
            raise FileNotFoundError(f"Tensor file {tensor_file} not found.")

        t = torch.arange(T + 1)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
        self.alpha_bar = f_t / f_t[0]
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar)

        if mu_ref is None:
            all_refs = [self.data_cache[f"ref{r.id}"].numpy() for r in self.records]
            all_refs = np.concatenate(all_refs, axis=0)
            self.mu_ref = torch.tensor(all_refs.mean(axis=0)).float()
        else:
            self.mu_ref = torch.tensor(mu_ref).float()

    def get_mu_ref(self):
        return self.mu_ref

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        r = self.records[idx]
        ref = self.data_cache[f"ref{r.id}"]         # (76, 6)
        delta = self.data_cache[f"delta{r.id}"]     # (76, 24)

        loss_mask = torch.isfinite(delta).float()

        ref = util.wrap(ref - self.mu_ref)       # (76, 6)

        t = torch.randint(0, self.T, (1,)).long()
        eps = torch.randn(delta.shape)
        x = delta * self.alpha_bar_sqrt[t] + eps * self.one_minus_alpha_bar_sqrt[t]
        x = util.wrap(x)

        return {"x": x, "t": t, "eps": eps, "ref": ref, "loss_mask": loss_mask}


def get_meta_data(tensor_file):
    data = torch.load(tensor_file)
    ids = sorted(set(k[3:] for k in data.keys() if k.startswith("ref")))
    meta = pd.DataFrame({"id": ids, "num_residues": [76] * len(ids)})
    meta.to_csv("meta.csv", index=False)


def parse_arguments(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="meta.csv")
    parser.add_argument("--tensor_file", type=str, default="/work/projects/ptao/ML_Protein_Dynamics/prodyna_ml/membersdata/chuanyexiong/project2_traj_generation/ubiquitin/real_data/50000frame_stride1000/data/conditioned_traj_dataset_5snapshots_wrapped.pt")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1000)
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def training(args_list: list):
    args = parse_arguments(args_list)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not os.path.exists(args.meta):
        get_meta_data(args.tensor_file)

    meta = pd.read_csv(args.meta).sample(frac=1.0, random_state=42)
    N = len(meta)
    train_meta, val_meta = meta.iloc[:int(0.9 * N)], meta.iloc[int(0.9 * N):]

    train_set = FoldingDiffDataset(meta=train_meta, tensor_file=args.tensor_file, T=args.timesteps)
    mu_ref = train_set.get_mu_ref().cpu().numpy().tolist()

    val_set = FoldingDiffDataset(meta=val_meta, tensor_file=args.tensor_file, T=args.timesteps, mu_ref=mu_ref)

    config = {
        "meta": args.meta,
        "tensor_file": args.tensor_file,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "mu_ref": mu_ref,
        "max_epochs": 10000,
    }

    wandb.login()
    wandb_logger = pl.loggers.WandbLogger(project="Proj2_Gen_ubi_Traj", name="conditional_diffusion_test5snapshots_05022025", config=config)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='foldingdiff-{epoch:04d}-{val_loss:.4f}', save_top_k=-1, every_n_epochs=50)

    trainer = pl.Trainer(accelerator="auto", devices='auto', max_epochs=config["max_epochs"], logger=wandb_logger, callbacks=[checkpoint_callback])
    model = FoldingDiff()
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    training(None)
