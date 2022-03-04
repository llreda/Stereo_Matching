import cv2
import numpy as np
import torch
import torchvision.utils as vutils
import math


def display_color_depth(epoch, step, writer, colors, pred_depth, gt_depth, mask, phase='Training', is_return_img=False, color_reverse=True):
    pred_depth = pred_depth[:, 2, :, :].unsqueeze(1).cpu()
    gt_depth = gt_depth[:, 2, :, :].unsqueeze(1).cpu()
    mask = mask[:, 2, :, :].unsqueeze(1).cpu()

    b, c, h, w = gt_depth.shape
    for i in range(b):
        if (gt_depth[i])[mask[i]].numel() < 200:
            min_val = 100
            max_val = 0
        else:
            min_val = torch.min((gt_depth[i])[mask[i]])
            max_val = torch.max((gt_depth[i])[mask[i]])
        min_cp = torch.min((pred_depth[i]))
        max_cp = torch.max((pred_depth[i]))
        min_val = min(min_val, min_cp)
        max_val = max(max_val, max_cp)
        gt_depth[i][mask[i]] = (gt_depth[i][mask[i]] - min_val) / (max_val - min_val)
        gt_depth[i][~mask[i]] = 0
        pred_depth[i] = (pred_depth[i] - min_val) / (max_val - min_val)

    nrow = math.ceil(b / 2)
    color_display = vutils.make_grid(colors * 0.5 + 0.5, nrow=nrow, normalize=False)
    color_display = np.moveaxis(color_display.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    pred_depth_display = vutils.make_grid(pred_depth, nrow=nrow, normalize=False)
    gt_depth_display = vutils.make_grid(gt_depth, nrow=nrow, normalize=False)
    mask_display = vutils.make_grid(mask, nrow=nrow, normalize=False)

    pred_depth_display = cv2.applyColorMap(
        np.uint8(255 * np.moveaxis(pred_depth_display.numpy(),
                                   source=[0, 1, 2], destination=[2, 0, 1])), cv2.COLORMAP_JET)
    gt_depth_display = cv2.applyColorMap(
        np.uint8(255 * np.moveaxis(gt_depth_display.numpy(),
                                   source=[0, 1, 2], destination=[2, 0, 1])), cv2.COLORMAP_JET)

    mask_display = np.moveaxis(mask_display.numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    for i in range(b):
        row = int(i / nrow)
        col = int(i % nrow)
        top = (row + 1) * 2 + row * h
        bottom = top + h
        left = (col + 1) * 2 + col * w
        right = left + w
        gt_depth_display[top:bottom, left:right, :][~ mask_display[top:bottom, left:right, :]] = 0

    if color_reverse:
        color_display = cv2.cvtColor(color_display, cv2.COLOR_BGR2RGB)
        pred_depth_display = cv2.cvtColor(pred_depth_display, cv2.COLOR_BGR2RGB)
        gt_depth_display = cv2.cvtColor(gt_depth_display, cv2.COLOR_BGR2RGB)

    if is_return_img:
        return color_display
    else:
        writer.add_image(phase + '/Images/epoch{}_color'.format(epoch), color_display, step, dataformats='HWC')
        writer.add_image(phase + '/Images/epoch{}_predicted_depth'.format(epoch), pred_depth_display, step, dataformats='HWC')
        writer.add_image(phase + '/Images/epoch{}_gt_depth'.format(epoch), gt_depth_display, step, dataformats='HWC')
        return


def display_color_disparity(epoch, step, writer, colors, pred_disparity, gt_disparity, mask, phase='Training', is_return_img=False, color_reverse=True):
    pred_disparity = pred_disparity.data.cpu()
    mask = (mask > 0.6)
    color_display = vutils.make_grid(colors * 0.5 + 0.5, normalize=False)   # 与数据标准化相反。make_grid可能是将batch分开了
    color_display = np.moveaxis(color_display.numpy(), source=[0, 1, 2], destination=[2, 0, 1])  # numpy与tensor的维度不一样

    disparity_display = vutils.make_grid(pred_disparity, normalize=False)
    gt_disparity_display = vutils.make_grid(gt_disparity, normalize=False)
    mask_display = vutils.make_grid(mask, normalize=False)

    b, c, h, w = gt_disparity.shape
    for i in range(b):
        left = (i + 1) * 2 + i * w
        right = left + w
        if (gt_disparity[i])[mask[i]].numel() < 200:
            min_val = 100
            max_val = 0
        else:
            min_val = torch.min((gt_disparity[i])[mask[i]])
            max_val = torch.max((gt_disparity[i])[mask[i]])

        min_cp = torch.min((pred_disparity[i]))
        max_cp = torch.max((pred_disparity[i]))
        min_val = min(min_val, min_cp)
        max_val = max(max_val, max_cp)
        gt_disparity_display[:, 2: (2 + h), left:right] = (gt_disparity_display[:, 2: (2 + h), left:right] - min_val) / (max_val - min_val)
        gt_disparity_display[:, 2: (2 + h), left:right][~ mask_display[:, 2: (2 + h), left:right]] = 0
        disparity_display[:, 2: (2 + h), left:right] = (disparity_display[:, 2: (2 + h), left:right] - min_val) / (
                    max_val - min_val)

    disparity_display = cv2.applyColorMap(
        np.uint8(255 * np.moveaxis(disparity_display.numpy(), source=[0, 1, 2], destination=[2, 0, 1])),
        cv2.COLORMAP_JET)

    gt_disparity_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(gt_disparity_display.numpy(), source=[0, 1, 2], destination=[2, 0, 1])), cv2.COLORMAP_JET)
    mask_display = np.moveaxis(mask_display.numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    for i in range(b):
        left = (i + 1) * 2 + i * w
        right = left + w
        (gt_disparity_display[2: (2 + h), left:right, :])[~ mask_display[2: (2 + h), left:right, :]] = 0

    if color_reverse:
        color_display = cv2.cvtColor(color_display, cv2.COLOR_BGR2RGB)
        disparity_display = cv2.cvtColor(disparity_display, cv2.COLOR_BGR2RGB)
        gt_disparity_display = cv2.cvtColor(gt_disparity_display, cv2.COLOR_BGR2RGB)

    if is_return_img:
        return color_display, disparity_display.astype(np.float32)/255.0
    else:
        writer.add_image(phase + '/Images/epoch{}_color'.format(epoch), color_display, step, dataformats='HWC')
        writer.add_image(phase + '/Images/epoch{}_predicted_disparity'.format(epoch), disparity_display, step, dataformats='HWC')
        writer.add_image(phase + '/Images/epoch{}_gt_disparity'.format(epoch), gt_disparity_display, step, dataformats='HWC')
        return