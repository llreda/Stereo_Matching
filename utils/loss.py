import torch.nn.functional as F


class Smooth_L1_XYZ_loss(object):

    def __call__(self, pred, target, mask):
        loss = F.smooth_l1_loss(pred[mask], target[mask], size_average=True)
        return loss


class Smooth_L1_Z_loss(object):

    def __call__(self, pred, target, mask):
        mask = mask[:, 2, :, :]
        pred = pred[:, 2, :, :]
        target = target[:, 2, :, :]
        loss = F.smooth_l1_loss(pred[mask], target[mask], size_average=True)
        return loss

class Smooth_reconstruction_loss(object):
    def __call__(self, warped, original, mask):
        loss = F.smooth_l1_loss(warped[mask], original[mask])
        return loss