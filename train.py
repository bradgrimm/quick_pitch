import pytorch_lightning as pl
import typer as typer
from pytorch_lightning.loggers import WandbLogger

from model import QuickPitch

app = typer.Typer()


@app.command()
def train(
        batch_size: int = 4,
        max_epochs: int = 1000,
        num_channels: int = 4,
        dilation_depth: int = 9,
        num_repeat: int = 2,
        kernel_size: int = 3,
        learning_rate: float = 3e-5,
):
    params = locals()
    model = QuickPitch(params)

    wandb_logger = WandbLogger(project="quick_pitch")
    trainer = pl.Trainer(logger=wandb_logger)
    trainer.fit(model)


if __name__ == '__main__':
    train()
