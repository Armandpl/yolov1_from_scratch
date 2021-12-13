import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import wandb

from network import YoloNetwork
from dataset import YoloCOCO
from loss import YoloLoss


class Yolo(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # setting up metrics
        # metrics = torchmetrics.MetricCollection([
        #     torchmetrics.MeanSquaredError(),
        #     torchmetrics.MeanAbsoluteError()
        # ])
        # self.train_metrics = metrics.clone(prefix='train/')
        # self.valid_metrics = metrics.clone(prefix='val/')
        # self.test_metrics = metrics.clone(prefix='test/')

        self.model = YoloNetwork()
        self.model.feature_extractor.train(False)

        # for param in self.model.feature_extractor.parameters():
        #    param.requires_grad = False

        self.loss = YoloLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        coords_loss, obj_loss, noobj_loss, classes_loss = self.loss(preds, targets)
        loss = coords_loss + obj_loss + noobj_loss + classes_loss

        # metrics = self.train_metrics(preds, targets)
        self.log_dict({
            "train/loss": loss,
            "train/coords_loss": coords_loss,
            "train/obj_loss": obj_loss,
            "train/noobj_loss": noobj_loss,
            "train/classes_loss": classes_loss
            }, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        # print("setting fe back to train mode")
        # for param in self.model.feature_extractor.parameters():
        #     param.requires_grad = True
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        coords_loss, obj_loss, noobj_loss, classes_loss = self.loss(preds, targets)
        loss = coords_loss + obj_loss + noobj_loss + classes_loss

        # metrics = self.valid_metrics(preds, targets)
        self.log_dict({"val/loss": loss}, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.learning_rate)
        return optimizer


class YoloDataModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train = YoloCOCO()
        self.val = YoloCOCO(
            root="./data/val2017/",
            annFile="./data/annotations/instances_val2017.json")

    def train_dataloader(self):
        return self.make_loader(self.train, True)

    def val_dataloader(self):
        return self.make_loader(self.val, False)

    def make_loader(self, dataset, shuffle):
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=True, num_workers=12)


if __name__ == "__main__":
    config = dict(
        epochs=10,
        architecture="resnet50",
        batch_size=128,
        learning_rate=1e-2
    )
    with wandb.init(project="yolov1", config=config, job_type="debug") as run:
        config = run.config

        dm = YoloDataModule(config.batch_size)
        yolo = Yolo(config)

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=1,
            max_epochs=config.epochs,
            log_every_n_steps=1
        )
        trainer.fit(yolo, dm)
