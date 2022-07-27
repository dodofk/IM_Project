import os
import torch
import argparse
import json
import hydra
from statistics import stdev, mean
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datamodules.components.cholect45_dataset import CholecT45Dataset, default_collate_fn
from models.cholec_baseline_module import TripletBaselineModule
from torchmetrics import Precision
from pprint import pprint

VALIDATION_VIDEOS = ["78", "43", "62", "35", "74", "1", "56", "4", "13"]
data_dir = "data/CholecT45/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="eval for video based mean and stdev",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        required=True,
        help="path to eval result file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="device for validation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for validaiton",
    )
    args = parser.parse_args()
    return args


@hydra.main(config_path="configs/")
def validation(cfg, args):

    device = args.device if torch.cuda.is_available() else "cpu"

    valid_record = dict()

    print("---- Loading Model ----")
    model = TripletBaselineModule.load_from_checkpoint(args.ckpt_path).to(device)
    model.eval()
    print("---- Finish Loading ----")

    for video in VALIDATION_VIDEOS:
        print(f"Video: {video}")
        dataset = CholecT45Dataset(
            img_dir=os.path.join(data_dir, "data", video),
            triplet_file=os.path.join(data_dir, "triplet", "{}.txt".format(video)),
            tool_file=os.path.join(data_dir, "instrument", "{}.txt".format(video)),
            verb_file=os.path.join(data_dir, "verb", "{}.txt".format(video)),
            target_file=os.path.join(data_dir, "target", "{}.txt".format(video)),
            split="dev",
            data_dir=data_dir,
            seq_len=2,
            channels=3,
            use_train_aug=False,
            triplet_class_arg="data/triplet_class_arg.npy",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_szie,
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

        with torch.no_grad():
            for batch in tqdm(dataloader):
                tool_logit, target_logit, verb_logit, triplet_logit = model(batch)

                triplet_map(triplet_logit, batch["triplet"].to(torch.int))
                tool_map(tool_logit, batch["tool"].to(torch.int))
                verb_map(verb_logit, batch["verb"].to(torch.int))
                target_map(target_logit, batch["target"].to(torch.int))

        valid_record[video] = {
            "triplet": triplet_map.compute().item(),
            "tool": tool_map.compute().item(),
            "verb": verb_map.compute().item(),
            "target": target_map.compute().item()
        }


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
    }

    with open(args.output_fname, "w") as f:
        json.dump(valid_record, f, sort_keys=True, indent=4)

    pprint(valid_record)


if __name__ == "__main__":
    args = parse_args()
    validation(args=args)








