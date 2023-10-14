import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import load_datasets
from quick_pitch import BasicPitch


class QuickPitch(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.basic_pitch = BasicPitch()
        self.params = params
        self.save_hyperparameters()

    def prepare_data(self):
        self.train_dataset, self.val_dataset = load_datasets()

    def configure_optimizers(self):
        return torch.optim.Adam(self.basic_pitch.parameters(), lr=self.params['learning_rate'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.params['batch_size'],
            num_workers=20,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params['batch_size'], num_workers=20)

    def forward(self, x):
        return self.basic_pitch(x)

    def training_step(self, batch, batch_idx):
        result = self._step(batch, batch_idx, 'train')
        result['loss'] = result['train_loss']
        self.log_dict(result)
        return result

    def validation_step(self, batch, batch_idx):
        result = self._step(batch, batch_idx, 'val')
        self.log_dict(result)
        return result

    def _step(self, batch, batch_idx, prefix):
        output = self.forward(batch['audio'])
        y, y_pred = batch['contour'], output['contour']
        print(y.shape, y_pred.shape)
        loss = F.mse_loss(y_pred, y)
        return {f"{prefix}_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}
