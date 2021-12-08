import math

import torchvision.datasets as dsets
from torchvision import transforms
import torch


class YoloCOCO(torch.utils.data.Dataset):
    def __init__(self, root="./data/train2017",
                 annFile="./data/annotations/instances_train2017.json",
                 n_classes=80, S=7):

        self.coco = dsets.CocoDetection(root=root, annFile=annFile)
        self.n_classes = n_classes
        self.S = S

        self.tfms = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((224, 224)),
                ])

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, label = self.coco.__getitem__(idx)
        im_w, im_h = img.size

        img = self.tfms(img)

        cell_side = 1/self.S
        targets_per_cell = torch.zeros((self.S, self.S, self.n_classes + 5))
        bboxes = [(x["bbox"], x["category_id"]) for x in label]

        # iterate over bboxes
        for bbox, class_id in bboxes:
            top_left_x, top_left_y, width, height = bbox
            center_x, center_y = \
                top_left_x + width/2, top_left_y + height/2
            # make bbox coordinates relative to the image size
            center_x, center_y, width, height = \
                top_left_x/im_w, top_left_y/im_h,\
                width/im_w, height/im_h

            # iterate over cells
            for idx in range(self.S**2):
                x_idx = math.floor(idx/self.S)
                y_idx = idx % self.S

                min_x, min_y, max_x, max_y = \
                    x_idx*cell_side, y_idx*cell_side,\
                    (x_idx+1)*cell_side, (y_idx+1)*cell_side

                # if current cell contains bbox
                if min_x < center_x < max_x and min_y < center_y < max_y:
                    # translate bbox coords from img ref to cell ref
                    new_center_x = (center_x - min_x) / cell_side
                    new_center_y = (center_y - min_y) / cell_side

                    new_w = width/cell_side
                    new_h = height/cell_side

                    # one hot encoded for class
                    targets_per_cell[x_idx, y_idx, class_id] = 1
                    targets_per_cell[x_idx, y_idx, -5] = 1  # probability
                    targets_per_cell[x_idx, y_idx, -4:] = \
                        [new_center_x, new_center_y, new_w, new_h]  # bbox

                    break

        return img, targets_per_cell.flatten()
