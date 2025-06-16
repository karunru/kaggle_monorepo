import numpy as np
import schedulefree
import torch
from augmentations.augmentation import get_default_transforms
from augmentations.strong_aug import get_strong_transforms
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn

from models.losses import FocalLossBCE, SoftAUCLoss  # noqa


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


class AttBlockV2(nn.Module):
    """Attention block for SED tasks."""

    def __init__(self, in_features, out_features, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (batch_size, channels, time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)

        if self.activation == "sigmoid":
            eps = 1e-6
            x = torch.clamp(x, eps, 1 - eps)
        return x

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        return None


class Model(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.build_model()
        self._criterion = eval(self.cfg.loss.name)(**self.cfg.loss.params)
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(
            OmegaConf.to_container(
                self.cfg,
                resolve=True,
            )
        )
        self.training_step_losses = []
        self.validation_step_losses = []

    def build_model(self):
        self.backbone = create_model(
            self.cfg.model.backbone,
            pretrained=True,
            num_classes=0,
            in_chans=self.cfg.model.in_chans,
            global_pool=self.cfg.model.pool_type,
        )
        num_features = self.backbone.num_features

        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_features, self.cfg.model.output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return f, out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        self.training_step_losses.append(loss.item())
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        self.validation_step_losses.append(loss.item())
        return {"pred": pred, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        images = self.transform["val"](images)
        embed, pred = self.forward(images)
        return pred, embed

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float()

        images = self.transform[mode](images)

        # if torch.rand(1)[0] < 0.5 and mode == "train":
        #     mix_images, target_a, target_b, lam = self.strong_transform(
        #         images, labels, **self.cfg.strong_transform.params
        #     )
        #     logits = self.forward(mix_images).squeeze(1)
        #     loss = self._criterion(logits, target_a) * lam + (
        #         1 - lam
        #     ) * self._criterion(logits, target_b)
        # else:
        #     logits = self.forward(images).squeeze(1)
        #     loss = self._criterion(logits, labels)
        embed, logits = self.forward(images)
        loss = self._criterion(logits, labels)

        pred = logits.detach().cpu()
        labels = labels.detach().cpu()
        return loss, pred, labels

    def on_train_epoch_end(self):
        self.__share_epoch_end(mode="train")
        self.training_step_losses.clear()

    def on_validation_epoch_end(self):
        self.__share_epoch_end(mode="val")
        self.validation_step_losses.clear()

    def __share_epoch_end(self, mode):
        metrics = np.mean(self.training_step_losses) if mode == "train" else np.mean(self.validation_step_losses)
        self.log(f"{mode}_loss", metrics)

    def configure_optimizers(self):
        if self.cfg.optimizer.opt == "radam_schedule_free":
            # Use RAdamScheduleFree optimizer
            optimizer = schedulefree.RAdamScheduleFree(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.get("betas", (0.9, 0.999)),
                eps=self.cfg.optimizer.get("eps", 1e-8),
                weight_decay=self.cfg.optimizer.get("weight_decay", 0),
                r=self.cfg.optimizer.get("r", 0.0),
                weight_lr_power=self.cfg.optimizer.get("weight_lr_power", 2.0),
                foreach=self.cfg.optimizer.get("foreach", True),
            )
            # Schedule-free optimizers don't need external schedulers
            return [optimizer]
        else:
            # Use regular optimizer with scheduler
            optimizer = create_optimizer_v2(self.parameters(), **self.cfg.optimizer)
            scheduler, _ = create_scheduler_v2(optimizer, **self.cfg.scheduler)

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

            return [optimizer], [scheduler_config]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=True)

    def on_train_start(self):
        # Set schedule-free optimizer to train mode
        if self.cfg.optimizer.opt == "radam_schedule_free":
            for optimizer in self.trainer.optimizers:
                optimizer.train()

    def on_validation_start(self):
        # Set schedule-free optimizer to eval mode
        if self.cfg.optimizer.opt == "radam_schedule_free":
            for optimizer in self.trainer.optimizers:
                optimizer.eval()

    def on_validation_end(self):
        # Set schedule-free optimizer back to train mode
        if self.cfg.optimizer.opt == "radam_schedule_free":
            for optimizer in self.trainer.optimizers:
                optimizer.train()
