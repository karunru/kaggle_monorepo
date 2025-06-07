import numpy as np
from augmentations.augmentation import get_default_transforms
from augmentations.strong_aug import get_strong_transforms
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn


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
            in_chans=1,
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
