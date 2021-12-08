import torch


def xywh2xyxy(xywh):
    x = xywh[..., 0]
    y = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]

    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2

    xyxy = torch.stack((x1, y1, x2, y2), dim=3)

    return xyxy

# def xyxy2xywh(xyxy):

# topxtopywh2xywh(topxtopywh):


def area(x1, y1, x2, y2):
    L = (x2 - x1).clamp(min=0)  # if they don't intersect
    w = (y2 - y1).clamp(min=0)
    return w*L


def IoU(box1, box2):
    box1 = xywh2xyxy(box1)
    box2 = xywh2xyxy(box2)

    # find intersection coords
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    # compute area
    intersection = area(x1, y1, x2, y2)

    # find union coords
    x1 = torch.min(box1[..., 0], box2[..., 0])
    y1 = torch.min(box1[..., 1], box2[..., 1])
    x2 = torch.max(box1[..., 2], box2[..., 2])
    y2 = torch.max(box1[..., 3], box2[..., 3])
    union = area(x1, y1, x2, y2)

    return torch.nan_to_num(intersection/union, nan=0.0)


class YoloLoss(torch.nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.lambda_coords = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.n_classes = 80
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, predictions, target):
        predictions =\
            predictions.reshape(-1, self.S, self.S, self.n_classes + self.B*5)

        ious_b1 = IoU(predictions[:, :, :, -4:], target[:, :, :, -4:])
        ious_b2 = IoU(predictions[:, :, :, -10:-5], target[:, :, :, -4:])
        ious = torch.cat([ious_b1.unsqueeze(0), ious_b2.unsqueeze(0)], dim=0)
        ious_max, ious_idx = torch.max(ious, dim=0)

        # is there an object in the cell
        obj = target[:, :, :, -5]

        # TODO this only works for B = 2

        # coords loss
        ious_idx_box_1 = torch.ones_like(ious_idx) - ious_idx
        ious_idx_box_2 = ious_idx

        coords_loss_box_1 =\
            self.loss(predictions[..., -4:-2], target[..., -4:-2]) +\
            torch.nan_to_num(
                self.loss(torch.sqrt(predictions[..., -2:]),
                          torch.sqrt(target[..., -2:])),
                nan=0.0
            )

        coords_loss_box_1 = torch.sum(coords_loss_box_1, dim=-1) *\
            ious_idx_box_1

        coords_loss_box_2 =\
            self.loss(predictions[..., -9:-7], target[..., -4:-2]) +\
            torch.nan_to_num(
                self.loss(torch.sqrt(predictions[..., -7:-5]),
                          torch.sqrt(target[..., -2:])),
                nan=0.0
            )

        coords_loss_box_2 = torch.sum(coords_loss_box_2, dim=-1) *\
            ious_idx_box_2

        coords_loss = self.lambda_coords * obj *\
            (coords_loss_box_1 + coords_loss_box_2)
        coords_loss = torch.sum(coords_loss, (1, 2))

        # obj loss
        obj_loss_b1 = self.loss(predictions[..., -5], target[..., -5]) *\
            ious_idx_box_1

        obj_loss_b2 = self.loss(predictions[..., -10], target[..., -5]) *\
            ious_idx_box_2

        obj_loss_raw = obj_loss_b1 + obj_loss_b2
        obj_loss = torch.sum(obj_loss_raw * obj, dim=(1, 2))

        # no obj loss
        noobj = torch.ones_like(obj) - obj
        noobj_loss = torch.sum(obj_loss_raw * noobj, dim=(1, 2)) *\
            self.lambda_noobj

        classes_loss =\
            torch.sum(
                self.loss(predictions[:, :, :, :-10], target[:, :, :, :-5]),
                dim=-1
            ) * obj
        classes_loss = torch.sum(classes_loss, dim=(1, 2))

        loss = coords_loss + obj_loss + noobj_loss + classes_loss
        # print(coords_loss, obj_loss, noobj_loss, classes_loss)
        return torch.sum(loss)
