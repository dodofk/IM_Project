import os
from typing import Any, List
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Precision
import ivtmetrics
from hydra.utils import get_original_cwd
import numpy as np
from pprint import pprint

assert timm.__version__ == "0.6.2.dev0", "Unsupport timm version"


class TripletAttentionModule(LightningModule):
    def __init__(
        self,
        temporal_cfg: DictConfig,
        optim: DictConfig,
        loss_weight: DictConfig,
        tool_component: DictConfig,
        target_tool_attention: DictConfig,
        use_pretrained: bool = True,
        backbone_model: str = "",
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
        self.train_tool_map = Precision(
            num_classes=self.class_num["tool"], average="macro"
        )
        self.train_verb_map = Precision(
            num_classes=self.class_num["verb"], average="macro"
        )
        self.train_target_map = Precision(
            num_classes=self.class_num["target"], average="macro"
        )
        self.train_triplet_map = Precision(
            num_classes=self.class_num["triplet"], average="macro"
        )
        self.valid_tool_map = Precision(
            num_classes=self.class_num["tool"], average="macro"
        )
        self.valid_verb_map = Precision(
            num_classes=self.class_num["verb"], average="macro"
        )
        self.valid_target_map = Precision(
            num_classes=self.class_num["target"], average="macro"
        )
        self.valid_triplet_map = Precision(
            num_classes=self.class_num["triplet"], average="macro"
        )

        assert (
            "vit" in backbone_model or "swin" in backbone_model
        ), "Only support using vision transformer based model"

        self.feature_extractor = timm.create_model(
            backbone_model,
            pretrained=use_pretrained,
            in_chans=3,
            num_classes=0,
        )

        self.tool_information = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                self.feature_extractor.num_features * tool_component.hidden_dim_size,
            ),
            getattr(nn, tool_component.activation_fn)(),
            nn.Dropout(p=tool_component.dropout_ratio),
            nn.Linear(
                self.feature_extractor.num_features * tool_component.hidden_dim_size,
                self.feature_extractor.num_features,
            ),
            getattr(nn, tool_component.activation_fn)(),
        )

        self.tool_head = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                self.class_num["tool"],
            ),
        )

        self.target_tool_attention = nn.MultiheadAttention(
            self.feature_extractor.num_features,
            batch_first=True,
            **target_tool_attention,
        )

        self.target_head = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                self.class_num["target"],
            ),
        )

        self.ts = nn.Sequential(
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
        )

        self.triplet_head = nn.Sequential(
            nn.Linear(
                temporal_cfg.hidden_size * self.temporal_direction(),
                self.class_num["triplet"],
            ),
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.triplet_map = self.contstruct_triplet_map()

        self.vit_dim = self.test_dim()

    def test_dim(self):
        self.feature_extractor.eval()
        x = torch.randn(1, 3, 224, 224)
        return self.feature_extractor.forward_features(x).shape[1]

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
            output[:, i, :, :] = self.feature_extractor.forward_features(
                x[:, i, :, :, :]
            )
        return output.to(self.device)

    def forward(self, x):
        output_tensor = torch.zeros(
            [x.shape[0], x.shape[1], self.vit_dim, self.feature_extractor.num_features]
        )
        feature = self.frames_feature_extractor(x, output_tensor)

        tool_seq_info = self.tool_information(feature[:, -1, :, :])

        tool_info = tool_seq_info.mean(dim=1)
        tool_logit = self.tool_head(tool_info)

        attn_output, _ = self.target_tool_attention(
            feature[:, -1, :, :],
            tool_seq_info,
            tool_seq_info,
            need_weights=False,
        )

        target_logit = self.target_head(attn_output.mean(dim=1))

        ts_feature, _ = self.ts(feature.mean(dim=2))
        verb_logit = self.verb_head((ts_feature[:, -1, :] + tool_info) / 2)
        triplet_logit = self.triplet_head((ts_feature[:, -1, :] + tool_info) / 2)

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
            self.hparams.loss_weight.tool_weight * tool_loss
            + self.hparams.loss_weight.target_weight * target_loss
            + self.hparams.loss_weight.verb_weight * verb_loss
            + self.hparams.loss_weight.triplet_weight * triplet_loss,
            tool_logit,
            target_logit,
            verb_logit,
            triplet_logit,
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        # self.train_recog_metric.update(
        #     batch["triplet"].cpu().numpy(),
        #     triplet_logit,
        # )
        self.train_tool_map(tool_logit, batch["tool"])
        self.train_target_map(target_logit, batch["target"])
        self.train_verb_map(verb_logit, batch["verb"])
        self.train_triplet_map(triplet_logit, batch["triplet"])
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/tool_mAP",
            self.train_tool_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/verb_mAP",
            self.train_verb_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/target_mAP",
            self.train_target_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/triplet_mAP",
            self.train_triplet_map,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_tool_map.reset()
        self.train_target_map.reset()
        self.train_verb_map.reset()
        self.train_triplet_map.reset()
        # ivt_result = self.train_recog_metric.compute_global_AP("ivt")
        # pprint(ivt_result["AP"])
        # self.log("train/ivt_mAP", ivt_result["mAP"])
        # self.log("train/i_mAP", self.train_recog_metric.compute_global_AP("i")["mAP"])
        # self.log("train/v_mAP", self.train_recog_metric.compute_global_AP("v")["mAP"])
        # self.log("train/t_mAP", self.train_recog_metric.compute_global_AP("t")["mAP"])

    def validation_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        self.valid_tool_map(tool_logit, batch["tool"])
        self.valid_target_map(target_logit, batch["target"])
        self.valid_verb_map(verb_logit, batch["verb"])
        self.valid_triplet_map(triplet_logit, batch["triplet"])

        self.log(
            "valid/tool_mAP",
            self.valid_tool_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "valid/verb_mAP",
            self.valid_verb_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "valid/target_mAP",
            self.valid_target_map,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "valid/triplet_mAP",
            self.valid_triplet_map,
            on_step=True,
            on_epoch=False,
        )

        tool_logit, target_logit, verb_logit, triplet_logit = (
            tool_logit.detach().cpu().numpy(),
            target_logit.detach().cpu().numpy(),
            verb_logit.detach().cpu().numpy(),
            triplet_logit.detach().cpu().numpy(),
        )

        self.valid_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            triplet_logit,
        )
        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        self.valid_tool_map.reset()
        self.valid_target_map.reset()
        self.valid_verb_map.reset()
        self.valid_triplet_map.reset()
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

        tool_logit, target_logit, verb_logit, triplet_logit = (
            tool_logit.detach().cpu().numpy(),
            target_logit.detach().cpu().numpy(),
            verb_logit.detach().cpu().numpy(),
            triplet_logit.detach().cpu().numpy(),
        )

        post_tool_logit, post_target_logit, post_verb_logit = (
            np.zeros([triplet_logit.shape[0], 100]),
            np.zeros([triplet_logit.shape[0], 100]),
            np.zeros([triplet_logit.shape[0], 100]),
        )

        for i in range(triplet_logit.shape[0]):
            for index, _triplet in enumerate(self.triplet_map):
                post_tool_logit[i][index] = tool_logit[i][_triplet[1]]
                post_verb_logit[i][index] = verb_logit[i][_triplet[2]]
                post_target_logit[i][index] = target_logit[i][_triplet[3]]

        self.test_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            # triplet_logit,
            triplet_logit + 0.4 * post_target_logit + 0.2 * post_verb_logit,
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