import lightning.pytorch as pl
from timm.utils import freeze, unfreeze


class FreezeUnfreezeCallback(pl.Callback):
    def __init__(self, freeze_epoch):
        self.freeze_epoch = freeze_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.freeze_epoch:
            freeze(pl_module.backbone)
        else:
            unfreeze(pl_module.backbone)
