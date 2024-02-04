import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class ConfiguredDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
            val_dataset: Dataset,
            batch_size: int,
    ):
        super().__init__()
        self.train = train_dataset
        self.test = test_dataset
        self.val = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=16
        )
