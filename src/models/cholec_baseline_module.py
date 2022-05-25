import os
from typing import Any, List
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import ivtmetrics
from hydra.utils import get_original_cwd
import numpy as np

from pprint import pprint


class TripletBaselineModule(LightningModule):
    def __init__(
        self,
        temporal_cfg: DictConfig,
        optim: DictConfig,
        # loss_weight: List = None,
        use_timm: bool = False,
        backbone_model: str = "resnet34",
        triplet_map: str = "./data/CholecT45/dict/maps.txt",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_recog_metric = ivtmetrics.Recognition(num_class=100)
        self.valid_recog_metric = ivtmetrics.Recognition(num_class=100)
        self.test_recog_metric = ivtmetrics.Recognition(num_class=100)

        self.class_num = {
            "tool": 6,
            "verb": 10,
            "target": 15,
            "triplet": 100,
        }

        self.feature_extractor = timm.create_model(
            backbone_model,
            pretrained=use_timm,
            in_chans=3,
            num_classes=0,
        )

        self.tool_head = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                self.class_num["tool"],
            ),
            # nn.Sigmoid(),
        )

        self.target_head = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                self.class_num["target"],
            ),
            # nn.Sigmoid(),
        )

        self.verb_ts = nn.Sequential(
            getattr(nn, temporal_cfg.type)(
                input_size=self.feature_extractor.num_features,
                hidden_size=temporal_cfg.hidden_size,
                num_layers=temporal_cfg.num_layers,
                bidirectional=temporal_cfg.bidirectional,
                batch_first=True,
            )
        )

        self.verb_head = nn.Sequential(
            nn.Linear(
                temporal_cfg.hidden_size * self.temporal_direction(),
                self.class_num["verb"],
            ),
            # nn.Sigmoid(),
        )

        self.triplet_ts = nn.Sequential(
            getattr(nn, temporal_cfg.type)(
                input_size=self.feature_extractor.num_features,
                hidden_size=temporal_cfg.hidden_size,
                num_layers=temporal_cfg.num_layers,
                bidirectional=temporal_cfg.bidirectional,
                batch_first=True,
            )
        )

        self.triplet_head = nn.Sequential(
            nn.Linear(
                temporal_cfg.hidden_size * self.temporal_direction(),
                self.class_num["triplet"],
            ),
            # nn.Sigmoid(),
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.triplet_map = self.contstruct_triplet_map()

    def contstruct_triplet_map(self):
        with open(os.path.join(get_original_cwd(), self.hparams.triplet_map), "r") as f:
            triplet_map = f.read().split("\n")[1:-2]

        ret = list()
        for triplet in triplet_map:
            ret.append(list(map(int, triplet.split(","))))

        return ret

    def temporal_direction(self):
        if (
            self.hparams.temporal_cfg.type is None
            or not self.hparams.temporal_cfg.bidirectional
        ):
            return 1
        else:
            return 2

    def frames_feature_extractor(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ):
        for i in range(0, x.shape[1]):
            output[:, i, :] = self.feature_extractor(x[:, i, :, :, :])
        return output.to(self.device)

    def forward(self, x):
        output_tensor = torch.zeros(
            [x.shape[0], x.shape[1], self.feature_extractor.num_features]
        )
        feature = self.frames_feature_extractor(x, output_tensor)

        tool_logit = self.tool_head(feature[:, -1, :])
        target_logit = self.target_head(feature[:, -1, :])
        verb_ts_feature, _ = self.verb_ts(feature)
        verb_logit = self.verb_head(verb_ts_feature[:, -1, :])
        triplet_ts_feature, _ = self.triplet_ts(feature)
        triplet_logit = self.triplet_head(triplet_ts_feature[:, -1, :])

        return tool_logit, target_logit, verb_logit, triplet_logit

    def step(self, batch: Any):
        """
        batch would a be a dict might contains the following things
        *image*: the frame image
        *action*: the action [Action type 0, Action type 1, Action type 3, Action type 4]
        *tool*: the tool [Tool 0, Tool 1, ..., Tool 6]
        *phase*: the phase [phase 0, ..., phase 6]

        ex:
        image = batch["image"]
        self.forward(image)

        return

        loss: the loss by the loss_fn
        preds: the pred by our model (i guess it would be sth like preds = torch.argmax(logits, dim=-1))
        y: correspond to the task it should be action or tool
        """
        tool_logit, target_logit, verb_logit, triplet_logit = self.forward(
            batch["image"]
        )
        tool_loss = self.criterion(tool_logit, batch["tool"])
        target_loss = self.criterion(target_logit, batch["target"])
        verb_loss = self.criterion(verb_logit, batch["verb"])
        triplet_loss = self.criterion(triplet_logit, batch["triplet"])
        return (
            tool_loss + target_loss + verb_loss + triplet_loss,
            tool_logit.detach().cpu().numpy(),
            target_logit.detach().cpu().numpy(),
            verb_logit.detach().cpu().numpy(),
            triplet_logit.detach().cpu().numpy(),
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        self.train_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            triplet_logit,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        ivt_result = self.train_recog_metric.compute_global_AP("ivt")
        pprint(ivt_result["AP"])
        self.log("train/ivt_mAP", ivt_result["mAP"])
        self.log("train/i_mAP", self.train_recog_metric.compute_global_AP("i")["mAP"])
        self.log("train/v_mAP", self.train_recog_metric.compute_global_AP("v")["mAP"])
        self.log("train/t_mAP", self.train_recog_metric.compute_global_AP("t")["mAP"])

    def validation_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        self.valid_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            triplet_logit,
        )
        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        ivt_result = self.valid_recog_metric.compute_global_AP("ivt")
        pprint(ivt_result["AP"])
        self.log(
            "valid/ivt_mAP",
            ivt_result["mAP"],
        )
        self.log("valid/i_mAP", self.valid_recog_metric.compute_global_AP("i")["mAP"])
        self.log("valid/v_mAP", self.valid_recog_metric.compute_global_AP("v")["mAP"])
        self.log("valid/t_mAP", self.valid_recog_metric.compute_global_AP("t")["mAP"])

    def test_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        # post_tool_logit, post_target_logit, post_verb_logit = (
        #     np.array([]),
        #     np.array([]),
        #     np.array([]),
        # )
        #
        # for _triplet in self.triplet_map:
        #     post_tool_logit = np.append(post_tool_logit, tool_logit[_triplet[1]])
        #     post_verb_logit = np.append(post_verb_logit, verb_logit[_triplet[2]])
        #     post_target_logit = np.append(post_target_logit, tool_logit[_triplet[3]])

        self.test_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            triplet_logit,
            # triplet_logit + 0.4 * post_tool_logit + 0.2 * post_verb_logit,
        )
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def test_epoch_end(self, outputs: List[Any]):
        ivt_result = self.test_recog_metric.compute_global_AP("ivt")
        pprint(ivt_result["AP"])
        self.log(
            "test/ivt_mAP",
            ivt_result["mAP"],
        )
        self.log("test/i_mAP", self.test_recog_metric.compute_global_AP("i")["mAP"])
        self.log("test/v_mAP", self.test_recog_metric.compute_global_AP("v")["mAP"])
        self.log("test/t_mAP", self.test_recog_metric.compute_global_AP("t")["mAP"])

    def on_epoch_end(self):
        self.train_recog_metric.reset()
        self.valid_recog_metric.reset()
        self.test_recog_metric.reset()

    def configure_optimizers(self):
        opt = getattr(torch.optim, self.hparams.optim.optim_name)(
            params=self.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay,
        )
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, self.hparams.optim.scheduler_name
        )(
            opt,
            **self.hparams.optim.scheduler,
        )
        return [opt], [lr_scheduler]
