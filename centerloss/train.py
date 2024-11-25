import lightning as L
from centerloss.modlit import CLModlit
from centerloss.data import CLDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import datetime
import wandb


def main():
    dm = CLDataModule(data_dir="./mnist", batch_size=128)

    net = CLModlit(0.01)

    # print(net)
    # exit(0)
    wandb_logger = WandbLogger(log_model=False, name='lenet', project='center loss')
    cp_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation loss",
        mode="min",
        dirpath="./model/",
        filename="lenet"
    )
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='cuda',
        logger=wandb_logger,
        callbacks=cp_callback,
        devices=1)
    trainer.fit(net, dm)


if __name__ == '__main__':
    main()
