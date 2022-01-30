import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import torchvision
import wandb

from network import YoloNetwork
from dataset import YoloCOCO
from loss import YoloLoss

with open("coco_classes.txt", "r") as f:
    coco_classes = f.read().splitlines()


def log_batch(images, preds, key, S=7, n_classes=91, B=2):
    im_w, im_h = images.size()[2:]
    cell_side = 1/S
    wandb_images = []
    for img, pred in zip(images[:32, ...], preds[:32, ...]):
        pred = pred.reshape(S, S, n_classes + 5*B)
        resized_boxes = []

        for i in range(2):
            if i == 0:
                boxes = pred[..., -4:]  # [P, cx, xy, w, h]*2
                conf = pred[..., -5]
            else:
                boxes = pred[..., -9:-5]
                conf = pred[..., -10]

            # [S, S, n_classes]
            clas = torch.argmax(pred[..., :-10], dim=2)

            # iterates over boxes, change the referential and size
            for x_idx in range(S):
                for y_idx in range(S):
                    cxcywh = boxes[x_idx, y_idx, :]
                    cx, cy = cxcywh[0]*cell_side + cell_side*x_idx,\
                        cxcywh[1]*cell_side + cell_side*y_idx
                    w, h = cxcywh[2]*cell_side, cxcywh[3]*cell_side

                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    box = [x1*im_w, y1*im_h, x2*im_w, y2*im_h,
                           conf[x_idx, y_idx], clas[x_idx, y_idx]]
                    resized_boxes.append(box)

        # nms
        box_data = []
        resized_boxes = torch.tensor(resized_boxes)  # [k, 6]
        b = resized_boxes[..., 0:4]
        scores = resized_boxes[..., 4]
        idxs = resized_boxes[..., 5]

        good_boxes_idxs = torchvision.ops.batched_nms(b, scores, idxs, 0.5)

        resized_boxes = resized_boxes[good_boxes_idxs].tolist()

        # create wandb image
        for minX, minY, maxX, maxY, conf, clas in resized_boxes:
            # label = names[c]
            box = {
                "position": {
                    "minX": int(minX),
                    "maxX": int(maxX),
                    "minY": int(minY),
                    "maxY": int(maxY)
                },
                "domain": "pixel",
                "class_id": int(clas),
                "box_caption": coco_classes[int(clas)],
                "scores": {
                    "conf": float(conf),
                }
            }

            box_data.append(box)
        boxes = {
            "predictions": {
                "box_data": box_data
            }
        }
        # create wandb bbox
        wandb_images.append(wandb.Image(img, boxes=boxes))
    wandb.log({f"{key}/images": wandb_images})


class Yolo(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = YoloNetwork()
        # self.model.freeze_feature_extractor(True)

        self.loss = YoloLoss()

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.should_log_batch = True

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        coords_loss, obj_loss, noobj_loss, classes_loss = \
            self.loss(preds, targets)
        loss = coords_loss + obj_loss + noobj_loss + classes_loss

        if self.should_log_batch:
            self.should_log_batch = False
            log_batch(images, preds, "train")

        self.log_dict({
            "train/loss": loss,
            "train/coords_loss": coords_loss,
            "train/obj_loss": obj_loss,
            "train/noobj_loss": noobj_loss,
            "train/classes_loss": classes_loss
            }, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        # after first epoch set feature extractor back to training mode
        # self.model.freeze_feature_extractor(False)
        pass

    def on_validation_start(self):
        self.should_log_batch = True

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        coords_loss, obj_loss, noobj_loss, classes_loss =\
            self.loss(preds, targets)
        loss = coords_loss + obj_loss + noobj_loss + classes_loss

        if self.should_log_batch:
            self.should_log_batch = False
            # log images + preds to wandb

            # create images to log them to wandb
            # TODO get that from self.model.
            log_batch(images, preds, "val")

        self.log_dict({
            "val/loss": loss,
            "val/coords_loss": coords_loss,
            "val/obj_loss": obj_loss,
            "val/noobj_loss": noobj_loss,
            "val/classes_loss": classes_loss
            }, on_step=False, on_epoch=True)

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
        epochs=200,
        architecture="resnet50",
        batch_size=128,
        learning_rate=1e-3
    )
    with wandb.init(project="yolov1", config=config, job_type="train") as run:
        config = run.config

        dm = YoloDataModule(config.batch_size)
        yolo = Yolo(config)
        # yolo.model.load_model("model:v1")

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=1,
            max_epochs=config.epochs,
            log_every_n_steps=1
        )
        try:
            trainer.fit(yolo, dm)
        except KeyboardInterrupt:
            pass

        yolo.model.save_feature_extractor()
        yolo.model.save_model()
