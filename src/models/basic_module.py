from typing import Any, List, Dict
from omegaconf import OmegaConf
from webbrowser import get

import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class BasicLitModule(LightningModule):
    def __init__(
        self,
        temporal_model: torch.nn.Module,
        mlp: object,
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

        # TODO: set up ResNet and LSTM, the following structure is a easy guide to implement
        # you could add any model component or write all model to the ./components folder
        # i think it might be more appropriate to write in components, so that we could use config to
        # easily change different model

        # DONE TODO: whether use the timm library or torch hub to load pretrained backbone model
        # should set a use_timm = true or false to determine
        self.feature_extractor = timm.create_model(
            "resnet34",
            pretrained=use_timm,
            in_chans = 1,
            # num_classes = ,
        )

        # not sure what name to use
        # DONE TODO: load the LSTM or other rnn based model it could be determine by config files
        # calling the falling line will error, since u didn't pass config to instantiate
        # self.temporal_model = instantiate(temporal_model)

        # DONE TODO: set up the final linear or with some activation functions
        self.mlp = nn.Sequential(
            nn.Linear(mlp.input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, mlp.output_size),
            nn.BatchNorm1d(mlp.output_size),
            nn.ReLU(),
        )

        # DONE TODO: choose the proper loss function, since i remember CE is better
        # use in one class for one instance ? or maybe i'm wrong
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        """
        return a tensor in size B * ? * (4 if task is action else 7)
        where B is the Batch Size ? according to the model setting
        """
        # DONE TODO: finish the foward part
        x = self.feature_extractor(x)
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
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

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
