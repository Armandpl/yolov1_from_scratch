import torch
from torchvision import ops


def batch_iou(pred_boxes, target_boxes):
    """
    takes in a batches of boxes (N, S, S, 4) in cx,cy,w,h format
    outputs ious (N, S, S)
    """
    bs = pred_boxes.size()[0]
    S = pred_boxes.size()[1]

    # flatten boxes bc torchvision ops takes list of boxes
    target_boxes = target_boxes.reshape((-1, S**2, 4))
    pred_boxes = pred_boxes.reshape((-1, S**2, 4))

    # convert boxes
    target_boxes = ops.box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    pred_boxes = ops.box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # compute ious
    ious = torch.zeros((bs, S**2), device=pred_boxes.device)
    for b_idx in range(bs):
        ious[b_idx] = ops.box_iou(target_boxes[b_idx], pred_boxes[b_idx]).diag()

    return ious

class YoloLoss(torch.nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.lambda_coords = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.n_classes = 80
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, predictions, target):
        predictions =\
            predictions.reshape(-1, self.S, self.S, self.n_classes + 5*self.B)

        target_boxes = target[..., -4:]
        pred_boxes_1 = predictions[..., -4:]
        pred_boxes_2 = predictions[..., -9:-5]

        # ious mask
        ious_b1, ious_b2 = batch_iou(pred_boxes_1, target_boxes),\
                           batch_iou(pred_boxes_2, target_boxes)

        ious = torch.stack([ious_b1, ious_b2], dim=0)
        _, ious_idx = torch.max(ious, dim=0)
        ious_idx = ious_idx.reshape((-1, self.S, self.S))

        # (N, S, S)
        ious_mask_box_1_ = torch.ones_like(ious_idx) - ious_idx
        ious_mask_box_2_ = ious_idx
        
        # obj mask: is there an object in the cell
        # (N, S, S)
        obj_mask_ = target[..., -5]

        # TODO this only works for B = 2

        # coords loss
        obj_mask = obj_mask_.unsqueeze(3).expand(-1, -1, -1, 4)
        ious_mask_box_1 = ious_mask_box_1_.unsqueeze(3).expand(-1, -1, -1, 4)
        ious_mask_box_2 = ious_mask_box_2_.unsqueeze(3).expand(-1, -1, -1, 4)

        # select box w/ biggest iou when there is object (else 0)
        pred_boxes = obj_mask * (ious_mask_box_1 * pred_boxes_1 + ious_mask_box_2 * pred_boxes_2) 

        dimensions = torch.sign(pred_boxes[..., -2:]) * torch.sqrt(
            torch.abs(pred_boxes[..., -2:]) + 1e-6  # numerical stability
        )
        target_dimensions = torch.sqrt(target_boxes[..., -2:])

        center_loss = self.mse(pred_boxes[..., :-3], target_boxes[..., :-3]) * self.lambda_coords
        dimensions_loss = self.mse(dimensions, target_dimensions) * self.lambda_coords
        coords_loss = center_loss + dimensions_loss

        # obj/noobj loss
        obj_pred = (predictions[..., -5]*ious_mask_box_1_ + predictions[..., -10]*ious_mask_box_2_)
        obj_loss = self.mse(obj_mask_ * obj_pred, target[..., -5])

        noobj_mask = torch.ones_like(obj_mask_) - obj_mask_
        noobj_loss = self.mse(noobj_mask * obj_pred, target[..., -5]) * self.lambda_noobj

        # classes loss
        obj_mask = obj_mask_.unsqueeze(3).expand(-1, -1, -1, self.n_classes)
        classes_loss = self.mse(predictions[..., :-10]*obj_mask, target[..., :-5])

        return coords_loss, obj_loss, noobj_loss, classes_loss
