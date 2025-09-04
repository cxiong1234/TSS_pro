import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from math import pi as PI
import util as util
import loss as loss


def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI


class LinearAnnealingLREpoch(LRScheduler):
    def __init__(self, optimizer, num_annealing_epochs, num_total_epochs, get_current_epoch_func):
        self.num_annealing_epochs = num_annealing_epochs
        self.num_total_epochs = num_total_epochs
        self.get_current_epoch_func = get_current_epoch_func
        super().__init__(optimizer)

    def get_lr(self):
        current_epoch = self.get_current_epoch_func()
        if current_epoch <= self.num_annealing_epochs:
            return [base_lr * current_epoch / self.num_annealing_epochs for base_lr in self.base_lrs]
        else:
            return [
                base_lr * (self.num_total_epochs - current_epoch) / (self.num_total_epochs - self.num_annealing_epochs)
                for base_lr in self.base_lrs
            ]


class RandomFourierFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 192, bias=False)
        nn.init.normal_(self.w.weight, std=2 * torch.pi)
        self.w.weight.requires_grad = False

    def forward(self, t):
        t = self.w(t.float())
        return torch.cat([torch.sin(t), torch.cos(t)], axis=-1)


class FoldingDiff(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.upscale = nn.Linear(24, 384)
        self.condition_proj = nn.Linear(24, 384)
        self.time_embed = RandomFourierFeatures()

        config = BertConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=384 * 2,
            max_position_embeddings=76,
            hidden_dropout_prob=0.1,
            position_embedding_type="relative_key",
        )
        self.encoder = BertEncoder(config)

        self.head = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 24),
        )

        self.criterion = loss.WrappedSmoothL1Loss(beta=0.1 * torch.pi)

    def forward(self, x, t, condition):
        x_embed = self.upscale(x)                          # (B, 76, 384)
        condition = util.wrap(condition)                   # (B, 76, 6)
        condition = condition.repeat(1, 1, 4)              # → (B, 76, 24)
        cond_embed = self.condition_proj(condition)        # → (B, 76, 384)

        time_embed = self.time_embed(t).unsqueeze(1)       # (B, 1, 384)
        h = x_embed + cond_embed + time_embed

        out = self.encoder(h)
        return self.head(out.last_hidden_state)

    def training_step(self, batch, batch_idx):
        x, t, eps, ref, loss_mask = batch["x"], batch["t"], batch["eps"], batch["ref"], batch["loss_mask"]
        out = self(x, t, condition=ref)
        loss = self.criterion(out * loss_mask, eps * loss_mask)

        self.log_dict({"train/loss": loss}, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        self.log("step", self.global_step, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t, eps, ref, loss_mask = batch["x"], batch["t"], batch["eps"], batch["ref"], batch["loss_mask"]
        out = self(x, t, condition=ref)
        loss = self.criterion(out * loss_mask, eps * loss_mask)

        self.log_dict({"val_loss": loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def get_current_epoch(self):
        return self.current_epoch

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearAnnealingLREpoch(
            optimizer,
            num_annealing_epochs=1000,
            num_total_epochs=5000,
            get_current_epoch_func=self.get_current_epoch
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
