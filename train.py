import pytorch_lightning as pl
import typer as typer
from pytorch_lightning.loggers import WandbLogger

from quick_pitch_training_module import QuickPitch

app = typer.Typer()


@app.command()
def train(
        batch_size: int = 16,
        max_epochs: int = 100,
        learning_rate: float = 5e-5,
):
    params = locals()
    model = QuickPitch(dict(params))

    wandb_logger = WandbLogger(project="quick_pitch")
    trainer = pl.Trainer(
        default_root_dir='/mnt/large/data/guitar/models/',
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
    )
    trainer.fit(model)


if __name__ == '__main__':
    train()
