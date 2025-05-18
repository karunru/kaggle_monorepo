import numpy as np
import torch
from omegaconf import OmegaConf
from overrides import overrides
from lightning.pytorch import LightningModule
from timm import create_model
from torch import nn
from augmentations.augmentation import get_default_transforms
from augmentations.strong_aug import get_strong_transforms


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
            in_chans=8,
        )
        num_features = self.backbone.num_features

        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_features, self.cfg.model.output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        f = self.emb(x)
        out = self.fc(f)
        return f, out

    def emb(self, x):
        return self.backbone(
            torch.permute(
                torch.cat((x, x), dim=1),
                (0, 3, 1, 2),
            ),
        )

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
        loss = self._criterion(logits.log(), labels)

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
        metrics = (
            np.mean(self.training_step_losses)
            if mode == "train"
            else np.mean(self.validation_step_losses)
        )
        self.log(f"{mode}_loss", metrics)

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=True)


class ChrisModel(Model):
    @overrides
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

    @overrides
    def emb(self, x):
        return self.backbone(self.__reshape_input(x))

    def __reshape_input(self, x: torch.Tensor):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # === Reshape (512,512,3) ===
        x = torch.cat([spectrograms, eegs], dim=2)

        if self.cfg.model.in_chans == 3:
            x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x


class WithTableModel(LightningModule):
    def __init__(self, cfg, batch_size=32):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.strong_transform = get_strong_transforms(self.cfg)
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        num_table_features = self.cfg.model.num_table_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features + num_table_features, self.cfg.model.output_dim),
        )

    def forward(self, image, features):
        f = self.backbone(image)
        f = torch.cat([f, features], dim=1)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, table_features, labels = batch
        images = self.transform["val"](images)
        out = self.forward(images, table_features).squeeze(1)
        if self.cfg.model.output_dim == 1:
            out = out.detach().cpu() * 100.0
        else:
            out = out.detach().cpu().sum(axis=1)
        return out

    def __share_step(self, batch, mode):
        images, table_features, labels = batch
        labels = labels.float()

        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = self.strong_transform(
                images, labels, **self.cfg.strong_transform.params
            )
            logits = self.forward(mix_images, table_features).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images, table_features).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.detach().cpu().sum(axis=1)
        labels = labels.detach().cpu().sum(axis=1)
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = nn.KLDivLoss()(preds, labels)
        self.log(f"{mode}_loss", metrics)

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        optimizer.zero_grad(set_to_none=False)
