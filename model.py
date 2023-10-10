import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_datasets


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            out = result[:, :, : -self.__padding]
            return out
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


def _downsample_stack():
    return nn.ModuleList([
        nn.Conv1d(88, 88, kernel_size=4, stride=2, padding=1)
        for i in range(8)
    ])


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.classifier = _downsample_stack()
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=88,
            kernel_size=1,
        )

        self.num_channels = num_channels

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        for downsample in self.classifier:
            out = downsample(out)
        return out.transpose(1, 2)


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class QuickPitch(pl.LightningModule):
    def __init__(self, params):
        super(QuickPitch, self).__init__()
        self.wavenet = WaveNet(
            num_channels=params["num_channels"],
            dilation_depth=params["dilation_depth"],
            num_repeat=params["num_repeat"],
            kernel_size=params["kernel_size"],
        )
        self.params = params

    def prepare_data(self):
        self.train_dataset, self.val_dataset = load_datasets()

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.params['learning_rate'])

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
        return self.wavenet(x)

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
        y_pred = self.forward(batch['audio'])
        y = batch['note']
        loss = F.mse_loss(y, y_pred)
        return {f"{prefix}_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


