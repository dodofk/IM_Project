from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.datamodules.components.cholect45_dataset import build_dataloader


class CholecT45DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seq_len: int = 8,
        channels: int = 3,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            split="train",
            data_dir=self.hparams.data_dir,
            seq_len=self.hparams.seq_len,
            channels=self.hparams.channels,
        )

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            split="dev",
            data_dir=self.hparams.data_dir,
            seq_len=self.hparams.seq_len,
            channels=self.hparams.channels,
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError