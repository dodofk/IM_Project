from typing import List, Dict, Tuple
import os

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision import transforms

import hydra
from hydra.utils import get_original_cwd


# todo: support for sequence data
class HeiCholeDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/HeiChole_data",
        split: str = "train",
    ) -> None:

        assert split in ["train", "dev"], "Invalid split"

        self.data_dir = data_dir

        self.df = pd.read_csv(
            os.path.join(
                get_original_cwd(),
                data_dir,
                f"{split}.csv",
            ),
        )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index) -> Dict:
        df_row = self.df.iloc[index]

        image = Image.open(
            os.path.join(
                get_original_cwd(),
                self.data_dir,
                f"HeiChole_{df_row['video_id']}",
                f"{df_row['image_id']}.jpg",
            )
        )

        image = self.transform(image)
        phase = df_row["phase"]

        return {
            "image": image,
            "phase": phase,
        }

    def __len__(self):
        return len(self.df)


def heichole_collate_fn(
    inputs: List,
) -> Dict:
    image = torch.stack([data["image"] for data in inputs])
    phase = torch.Tensor([data["phase"] for data in inputs]).to(torch.long)
    return {
        "image": image,
        "phase": phase,
    }


def build_heichole_dataloader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    split: str,
    data_dir: str,
) -> DataLoader:
    assert split in ["train", "dev"], "Invalid Split"

    dataset = HeiCholeDataset(
        split=split,
        data_dir=data_dir,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=heichole_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True if split == "train" else False,
    )


@hydra.main(config_path=None)
def test_dataset(cfg) -> None:
    dataset = HeiCholeDataset(
        data_dir="../../../../slue-toolkit/data/slue-voxpopuli/",
        split="fine-tune",
    )
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    test_dataset()