from typing import Any, List, Dict
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import F1Score


class BasicLitModule(LightningModule):
    def __init__(
        self,
        temporal_cfg: DictConfig,
        mlp: DictConfig,
        optim: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        task: str = "tool",
        use_timm: bool = False,
    ):
        super().__init__()

        """
        First implement two model which could handle tool detection or action detection, and could be choose 
        by basic.yaml config about task (self.hparams.task)
        """

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.feature_extractor = timm.create_model(
            "resnet34",
            pretrained=use_timm,
            in_chans=3,
            num_classes=0,
        )

        # todo: not yet support for tcn, or rnn-based model
        if temporal_cfg.type is not None:
            self.temporal_model = getattr(nn, temporal_cfg.type)(
                input_size=self.feature_extractor.num_features,
                hidden_size=temporal_cfg.hidden_size,
                num_layers=temporal_cfg.num_layers,
                bidirectional=temporal_cfg.bidirectional,
            )

        self.mlp = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features * self.temporal_direction(),
                mlp.hidden_size,
            ),
            nn.BatchNorm1d(mlp.hidden_size),
            nn.ReLU(),
            nn.Linear(mlp.hidden_size, self.num_class()),
        )

        if task in ["phase"]:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif task in ["tool", "action"]:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_f1 = F1Score()
        self.val_f1 = F1Score()

        # for logging best so far validation accuracy
        self.val_f1_best = MaxMetric()

    def num_class(self):
        task_class = {
            "tool": 7,
            "phase": 7,
            "action": 4,
        }
        return task_class.get(self.hparams.task)

    def temporal_direction(self):
        if (
            self.hparams.temporal_cfg.type is None
            or not self.hparams.temporal_cfg.bidirectional
        ):
            return 1
        else:
            return 2

    def forward(self, x):

        x = self.feature_extractor(x)

        if self.hparams.temporal_cfg.type is not None:
            x, _ = self.hparams.temporal_model(x)
        x = self.mlp(x)

        return x

    def step(self, batch: Any):
        """
        batch would a be a dict might contains the following things
        *image*: the frame image
        *action*: the action [Action type 0, Action type 1, Action type 3, Action type 4]
        *tool*: the tool [Tool 0, Tool 1, ..., Tool 6]

        ex:
        image = batch["image"]
        self.forward(image)

        return

        loss: the loss by the loss_fn
        preds: the pred by our model (i guess it would be sth like preds = torch.argmax(logits, dim=-1))
        y: correspond to the task it should be action or tool
        """

        # TODO: finish the step part and choose the proper loss function for multi-classification
        # print(batch[self.hparams.task])
        # x, y = batch["image"], batch[self.hparams.task]
        logits = self.forward(batch["image"])
        loss = self.criterion(logits, batch[self.hparams.task])
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, batch[self.hparams.task]

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

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
        self.test_f1.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        # change the optimizer could be determine by config files,
        # the following is the code i guess it should work

        return getattr(torch.optim, self.hparams.optim)(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
