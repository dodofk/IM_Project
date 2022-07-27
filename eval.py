import os
import torch
import json
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from statistics import stdev, mean
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import ivtmetrics

from torchmetrics import Precision
from pprint import pprint
from src.datamodules.components.cholect45_dataset import (
    CholecT45Dataset,
    default_collate_fn,
)
from src.models.cholec_baseline_module import TripletBaselineModule


VALIDATION_VIDEOS = ["78", "43", "62", "35", "74", "01", "56", "04", "13"]


@hydra.main(config_path="configs/", config_name="eval.yaml")
def validation(args):
    data_dir = os.path.join(get_original_cwd(), "data/CholecT45/")
    device = args.device if torch.cuda.is_available() else "cpu"

    valid_record = dict()

    print("---- Loading Model ----")
    model = TripletBaselineModule.load_from_checkpoint(
        os.path.join(get_original_cwd(), args.ckpt_path)
    ).to(device)
    model.eval()
    print("---- Finish Loading ----")
    global_ivt_metric = ivtmetrics.Recognition(num_class=100)
    global_ivt_metric.reset_global()

    for video in VALIDATION_VIDEOS:
        print(f"Video: {video}")
        dataset = CholecT45Dataset(
            img_dir=os.path.join(data_dir, "data", f"VID{video}"),
            triplet_file=os.path.join(data_dir, "triplet", "VID{}.txt".format(video)),
            tool_file=os.path.join(data_dir, "instrument", "VID{}.txt".format(video)),
            verb_file=os.path.join(data_dir, "verb", "VID{}.txt".format(video)),
            target_file=os.path.join(data_dir, "target", "VID{}.txt".format(video)),
            split="dev",
            data_dir=data_dir,
            seq_len=2,
            channels=3,
            use_train_aug=False,
            triplet_class_arg=os.path.join(
                get_original_cwd(), "data/triplet_class_arg.npy"
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=default_collate_fn,
            num_workers=0,
            shuffle=False,
        )
        triplet_map = Precision(
            num_classes=model.class_num["triplet"],
            average="macro",
        )
        tool_map = Precision(
            num_classes=model.class_num["tool"],
            average="macro",
        )
        verb_map = Precision(
            num_classes=model.class_num["verb"],
            average="macro",
        )
        target_map = Precision(
            num_classes=model.class_num["target"],
            average="macro",
        )
        ivt_metric = ivtmetrics.Recognition(num_class=100)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                tool_logit, target_logit, verb_logit, triplet_logit = model(
                    batch["image"].to(args.device)
                )

                triplet_map(triplet_logit.to("cpu"), batch["triplet"].to(torch.int))
                tool_map(tool_logit.to("cpu"), batch["tool"].to(torch.int))
                verb_map(verb_logit.to("cpu"), batch["verb"].to(torch.int))
                target_map(target_logit.to("cpu"), batch["target"].to(torch.int))

                tool_logit, target_logit, verb_logit, triplet_logit = (
                    softmax(tool_logit, dim=-1).detach().cpu().numpy(),
                    softmax(target_logit, dim=-1).detach().cpu().numpy(),
                    softmax(verb_logit, dim=-1).detach().cpu().numpy(),
                    softmax(triplet_logit, dim=-1).detach().cpu().numpy(),
                )

                post_tool_logit, post_target_logit, post_verb_logit = (
                    np.zeros([triplet_logit.shape[0], 100]),
                    np.zeros([triplet_logit.shape[0], 100]),
                    np.zeros([triplet_logit.shape[0], 100]),
                )

                for i in range(triplet_logit.shape[0]):
                    for index, _triplet in enumerate(model.triplet_map):
                        post_tool_logit[i][index] = tool_logit[i][_triplet[1]]
                        post_verb_logit[i][index] = verb_logit[i][_triplet[2]]
                        post_target_logit[i][index] = target_logit[i][_triplet[3]]

                combined_triplet_logit = triplet_logit + 0.4 * post_target_logit + 0.6 * post_verb_logit
                ivt_metric.update(
                    batch["triplet"].cpu().numpy(),
                    combined_triplet_logit,
                )
                global_ivt_metric.update(
                    batch["triplet"].cpu().numpy(),
                    combined_triplet_logit,
                )

        valid_record[video] = {
            "triplet": triplet_map.compute().item(),
            "tool": tool_map.compute().item(),
            "verb": verb_map.compute().item(),
            "target": target_map.compute().item(),
            "i_mAP": ivt_metric.compute_global_AP("i")["mAP"],
            "v_mAP": ivt_metric.compute_global_AP("i")["mAP"],
            "t_mAP": ivt_metric.compute_global_AP("v")["mAP"],
            "ivt_mAP": ivt_metric.compute_global_AP("t")["mAP"],
        }
        global_ivt_metric.video_end()

    valid_record["overall"] = {
        "triplet": {
            "mean": mean([record["triplet"] for record in valid_record.values()]),
            "stdev": stdev([record["triplet"] for record in valid_record.values()]),
        },
        "tool": {
            "mean": mean([record["tool"] for record in valid_record.values()]),
            "stdev": stdev([record["tool"] for record in valid_record.values()]),
        },
        "verb": {
            "mean": mean([record["verb"] for record in valid_record.values()]),
            "stdev": stdev([record["verb"] for record in valid_record.values()]),
        },
        "target": {
            "mean": mean([record["target"] for record in valid_record.values()]),
            "stdev": stdev([record["target"] for record in valid_record.values()]),
        },
        "i_mAP": {
            "mean": mean([record["i_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["i_mAP"] for record in valid_record.values()]),
        },
        "v_mAP": {
            "mean": mean([record["v_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["v_mAP"] for record in valid_record.values()]),
        },
        "t_mAP": {
            "mean": mean([record["t_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["t_mAP"] for record in valid_record.values()]),
        },
        "ivt_mAP": {
            "mean": mean([record["ivt_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["ivt_mAP"] for record in valid_record.values()]),
        },
        "g_i_mAP": global_ivt_metric.compute_global_AP("i")["mAP"],
        "g_v_mAP": global_ivt_metric.compute_global_AP("v")["mAP"],
        "g_t_mAP": global_ivt_metric.compute_global_AP("t")["mAP"],
        "g_ivt_mAP": global_ivt_metric.compute_global_AP("ivt")["mAP"],
        # "g_ivt_detail": list(global_ivt_metric.compute_global_AP("ivt")["AP"]),
    }

    with open(os.path.join(get_original_cwd(), args.output_fname), "w") as f:
        json.dump(valid_record, f, sort_keys=True, indent=4)

    pprint(valid_record)


if __name__ == "__main__":
    validation()
