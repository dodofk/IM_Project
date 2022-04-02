from typing import Any, List, Dict
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import F1Score


class BaseClassificationModele(LightningModule):
    """
    Basic module for classification to avoid duplicate code
    """

    def __init__(self):
        super().__init__()
        self.train_f1 = F1Score()
        self.val_f1 = F1Score()

        # for logging best so far validation accuracy
        self.val_f1_best = MaxMetric()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/f1", f1, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        f1 = self.val_f1.compute()  # get val accuracy from current epoch
        self.val_f1_best.update(f1)
        self.log(
            "val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_f1.reset()
        self.val_f1.reset()
