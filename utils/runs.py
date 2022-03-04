import argparse
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import datetime
import os, os.path
import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter
import utils.loss
from dataloader import SCARED_dataset as MyD
from models import basic, stackhourglass, submodule, gwc, constancy
from utils import display, metrics
import utils.loss as Myloss
import logging, logging.config


def get_model(config):
    assert config.model in ['stackhourglass', 'basic', 'constancy', 'gwc_g', 'gwc_gc'], f'no {config.model} model exist'
    if config.model == 'stackhourglass':
        model = stackhourglass.PSMNet(int(config.maxdisp))
    elif config.model == 'basic':
        model = basic.PSMNet(int(config.maxdisp))
    elif config.model == 'constancy':
        model = constancy.Feature_Constancy(int(config.maxdisp))
    elif config.model == 'gwc_g':
        model = gwc.GwcNet_G(int(config.maxdisp))
    elif config.model == 'gwc_gc':
        model = gwc.GwcNet_GC(int(config.maxdisp))
    if config.cuda:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model.cuda()

    epoch = 0
    step = 0
    logger = logging.getLogger('consolelog')
    if config.loadmodel is not None:
        if Path(config.loadmodel).exists():
            logger.info(f"Loading {config.loadmodel} .................")
            state_dict = torch.load(config.loadmodel)
            model.load_state_dict(state_dict['state_dict'])
            epoch = state_dict['epoch']
            step = state_dict['step']
            logger.info(f'Restored model, epoch {epoch}')
        else:
            logger.info("No trained model detected")
            exit()

    config.epoch = epoch
    config.step = step
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    if epoch > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=epoch)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    return model, optimizer, scheduler


def get_valid_model(config):
    assert config.model in ['stackhourglass', 'basic', 'constancy', 'gwc_g', 'gwc_gc'], f'no {config.model} model exist'
    if config.model == 'stackhourglass':
        model = stackhourglass.PSMNet(int(config.maxdisp))
    elif config.model == 'basic':
        model = basic.PSMNet(int(config.maxdisp))
    elif config.model == 'constancy':
        model = constancy.Feature_Constancy(int(config.maxdisp))
    elif config.model == 'gwc_g':
        model = gwc.GwcNet_G(int(config.maxdisp))
    elif config.model == 'gwc_gc':
        model = gwc.GwcNet_GC(int(config.maxdisp))
    if config.cuda:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model.cuda()

    epoch = 0
    step = 0
    logger = logging.getLogger('consolelog')
    if Path(config.loadmodel).exists():
        logger.info(f"Loading {config.loadmodel} .................")
        state_dict = torch.load(config.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        epoch = state_dict['epoch']
        step = state_dict['step']
        logger.info(f'Restored model, epoch {epoch}')
    else:
        logger.info("No trained model detected")
        exit()
    config.epoch = epoch
    config.step = step
    return model


def train(model, optimizer, criterion, sample, config):
    model.train()
    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    depth = sample['depth']  # [B, 3, H, W]
    mask = sample['mask']
    Q = sample['Q']
    if config.cuda:
        left, right, depth, mask, Q = left.cuda(), right.cuda(), depth.cuda(), mask.cuda(), Q.cuda()

    b, c, h, w = depth.shape
    count = b
    total = h * w * c
    for i in range(b):
        if mask[i].int().sum() < total * config.least_percent_of_valid:
            mask[i].fill_(False)
            count -= 1
    if count < 1:             # Too few valid supervisory signals
        return -1, -1, -1

    optimizer.zero_grad()
    if config.model == 'stackhourglass':
        output1, output2, output3 = model(left, right)

        output1 = F.upsample(output1, [h, w], mode='bilinear') * config.downsampling
        output2 = F.upsample(output2, [h, w], mode='bilinear') * config.downsampling
        output3 = F.upsample(output3, [h, w], mode='bilinear') * config.downsampling

        depth1 = submodule.reprojection()(output1.squeeze(1), Q)
        depth2 = submodule.reprojection()(output2.squeeze(1), Q)
        depth3 = submodule.reprojection()(output3.squeeze(1), Q)
        # loss = 0.5 * F.smooth_l1_loss(depth1[mask_left], rec_left_gt[mask_left]) + 0.7 * F.smooth_l1_loss(depth2[mask_left], rec_left_gt[mask_left]) + F.smooth_l1_loss(depth3[mask_left], rec_left_gt[mask_left])

        loss = 0.5 * criterion(depth1, depth, mask) + 0.7 * criterion(depth2, depth, mask) + \
               criterion(depth3, depth, mask)

    elif config.model == 'basic':
        output3 = model(left, right)
        output3 = F.upsample(output3, [h, w], mode='bilinear') * config.downsampling
        depth3 = submodule.reprojection()(output3.squeeze(1), Q)
        loss = criterion(depth3, depth, mask)

    elif config.model == 'constancy':
        output1, output2, output3, constancy_left, constancy_right = model(left, right)
        warp = submodule.warp_feature('rl', 8)
        warped_left1, mask1 = warp(output1, constancy_right)
        warped_left2, mask2 = warp(output2, constancy_right)
        warped_left3, mask3 = warp(output3, constancy_right)

        output1 = F.upsample(output1, [h, w], mode='bilinear') * config.downsampling
        output2 = F.upsample(output2, [h, w], mode='bilinear') * config.downsampling
        output3 = F.upsample(output3, [h, w], mode='bilinear') * config.downsampling

        depth1 = submodule.reprojection()(output1.squeeze(1), Q)
        depth2 = submodule.reprojection()(output2.squeeze(1), Q)
        depth3 = submodule.reprojection()(output3.squeeze(1), Q)

        constancy_loss = Myloss.Smooth_reconstruction_loss()
        loss = 0.5 * criterion(depth1, depth, mask) + 0.7 * criterion(depth2, depth, mask) + \
               criterion(depth3, depth, mask) + 0.3 * constancy_loss(warped_left1, constancy_left, mask1) + \
               0.4 * constancy_loss(warped_left2, constancy_left, mask2) + \
               0.5 * constancy_loss(warped_left3, constancy_left, mask3)
    elif config.model == 'gwc_g' or config.model == 'gwc_gc':
        output0, output1, output2, output3 = model(left, right)
        output0 = F.upsample(output0, [h, w], mode='bilinear') * config.downsampling
        output1 = F.upsample(output1, [h, w], mode='bilinear') * config.downsampling
        output2 = F.upsample(output2, [h, w], mode='bilinear') * config.downsampling
        output3 = F.upsample(output3, [h, w], mode='bilinear') * config.downsampling

        depth0 = submodule.reprojection()(output0.squeeze(1), Q)
        depth1 = submodule.reprojection()(output1.squeeze(1), Q)
        depth2 = submodule.reprojection()(output2.squeeze(1), Q)
        depth3 = submodule.reprojection()(output3.squeeze(1), Q)
        # loss = 0.5 * F.smooth_l1_loss(depth1[mask_left], rec_left_gt[mask_left]) + 0.7 * F.smooth_l1_loss(depth2[mask_left], rec_left_gt[mask_left]) + F.smooth_l1_loss(depth3[mask_left], rec_left_gt[mask_left])

        loss = 0.5 * criterion(depth0, depth, mask) + 0.5 * criterion(depth1, depth, mask) + 0.7 * criterion(depth2, depth, mask) + \
               criterion(depth3, depth, mask)

    loss.backward()
    optimizer.step()
    # display.display_color_disparity_depth(step, writer, ds_left, output3.unsqueeze(1), depth3, is_return_img=False)
    return loss.item(), count, depth3


def Test(model, criterion, sample, config):
    model.eval()
    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    depth = sample['depth']  # [B, 3, H, W]
    mask = sample['mask']
    Q = sample['Q']
    if config.cuda:
        left, right, depth, mask, Q = left.cuda(), right.cuda(), depth.cuda(), mask.cuda(), Q.cuda()

    b, c, h, w = depth.shape
    count = b
    total = h * w * c
    for i in range(b):
        if mask[i].int().sum() < total * config.least_percent_of_valid:
            mask[i].fill_(False)
            count -= 1
    if count < 1:             # Too few valid supervisory signals
        return -1, -1, -1

    loss = 0
    with torch.no_grad():
        output3 = model(left, right)
        output3 = F.upsample(output3, [h, w], mode='bilinear') * config.downsampling
        depth3 = submodule.reprojection()(output3.squeeze(1), Q)
        loss = criterion(depth3, depth, mask)
    depth3 = depth3.cpu()
    depth = depth.cpu()
    mask = mask.cpu()
    loss = loss.item()
    error = metrics.evaluate_error(depth3, depth, mask)
    error['loss'] = loss
    return error, count, depth3


def run(config):
    """
    train and validate a model
    :param config:
    :return:
    """

    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)

    # logging file, visualization, checkpoints save in the same dir(config.savemodel)
    logger = logging.getLogger('filelog')
    consolelogger = logging.getLogger('consolelog')
    currentDT = datetime.datetime.now()
    log_root = Path(config.savemodel) / "exp_{}_{}_{}_valid_{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour, config.validation_sets)
    if not log_root.exists():
        log_root.mkdir()
    writer = SummaryWriter(logdir=str(log_root))
    logger.info(f"***************************** Tensorboard visualization at {os.path.abspath(log_root)}")

    # get model and optimizer
    model, optimizer, scheduler = get_model(config)
    logger.info(f'***************************** Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    # prepare dataloader
    Traindata = MyD.SCARED_Dadaset(config, phase='train')
    TrainImgLoader = torch.utils.data.DataLoader(dataset=Traindata, batch_size=config.batch_size, shuffle=True,
                                                 drop_last=False)
    Validationdata = MyD.SCARED_Dadaset(config, phase='validation')
    ValidationImgLoader = torch.utils.data.DataLoader(dataset=Validationdata, batch_size=config.batch_size * 2,
                                                      shuffle=False,drop_last=False)
    # loss function
    criterion = Myloss.Smooth_L1_XYZ_loss()

    # start training
    start_full_time = time.time()
    start = config.epoch
    step = config.step
    display_interval = 50
    for epoch in range(start + 1, config.epochs + 1):
        total_loss = 0
        total_count = 0
        mean_loss = 0
        local_step = 0
        tq = tqdm.tqdm(total=len(TrainImgLoader) * config.batch_size, dynamic_ncols=True)
        tq.set_description('Training Epoch {}'.format(epoch))
        ## training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            loss, count, depth = train(model, optimizer, criterion, sample, config)
            if count < 0:
                continue
            else:
                total_loss += (loss * count)
                total_count += count
                mean_loss = total_loss / total_count

            step += 1
            local_step += 1
            tq.update(config.batch_size)
            tq.set_postfix({'avg_loss': mean_loss, 'curr_loss': loss})
            writer.add_scalar('Training/loss', mean_loss, step)
            if local_step % display_interval == 0:
                display.display_color_depth(epoch, local_step, writer, sample['left'], depth, sample['depth'],
                                            sample['mask'], phase='Training', is_return_img=False, color_reverse=True)
            # if local_step > 2:
            #     break
        tq.close()
        logger.info(f'***************************** epoch{epoch} has finished at step{step}')
        logger.info(f"***************************** mean_loss = {mean_loss}....................")

        savefilename = config.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'step': step,
            'state_dict': model.state_dict(),
            'train_loss': mean_loss,
        }, savefilename)
        writer.export_scalars_to_json(str(log_root / ('all_scalars_' + str(epoch) + '.json')))
        consolelogger.info(f"epoch{epoch} training finished and checkpoint.tar has been saved.")

        ##validation##
        total_error = {'loss': 0, 'MAE': 0, 'MAE_XYZ': 0, 'REL': 0, 'BAD1SCORE': 0, 'BAD2SCORE': 0, 'BAD5SCORE': 0, 'BAD10SCORE': 0}
        total_count = 0
        mean_error = 0
        local_step = 0
        tq = tqdm.tqdm(total=len(ValidationImgLoader) * config.batch_size, dynamic_ncols=True)
        tq.set_description('Validation Epoch {}'.format(epoch))
        with torch.no_grad():
            for batch_idx, sample in enumerate(ValidationImgLoader):
                error, count, depth = Test(model, criterion, sample, config)
                if count < 0:
                    continue
                else:
                    total_count += count
                    mean_error = metrics.avg_error(total_error, error, total_count, count)

                tq.update(config.batch_size)
                tq.set_postfix({'Valid_avg': mean_error['loss'], 'Valid_cur': error['loss']})
                local_step += 1
                writer.add_scalar('Validation/epoch{}_loss'.format(epoch), error['loss'], local_step)
                writer.add_scalar('Validation/epoch{}_MAE'.format(epoch), error['MAE'], local_step)
                # if local_step > 2:
                #     break
                if local_step % display_interval == 0:
                    display.display_color_depth(epoch, local_step, writer, sample['left'], depth, sample['depth'],
                                                sample['mask'], phase='Validation',is_return_img=False, color_reverse=True)
            logger.info(f'***************************** epoch{epoch} validation has finished, the metrics as show below:')
            for item, value in mean_error.items():
                logger.info(f'##################### {item} = {value}....................')
                writer.add_scalar(f'Validation/{item}', value, epoch)
        tq.close()

    logger.info('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


def TEST(config):

    # logging file, visualization, checkpoints save in the same dir(config.savemodel)
    logger = logging.getLogger('filelog')
    consolelogger = logging.getLogger('consolelog')
    currentDT = datetime.datetime.now()
    log_root = Path(config.savedir) / "exp_{}_{}_{}_test{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour, config.test_sets)
    if not log_root.exists():
        log_root.mkdir()
    writer = SummaryWriter(logdir=str(log_root))
    logger.info(f"***************************** Tensorboard visualization at {os.path.abspath(log_root)}")

    # get model
    model = get_valid_model(config)
    logger.info(f'***************************** Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    # prepare dataloader
    Testdata = MyD.SCARED_Dadaset(config, phase='test')
    TestImgLoader = torch.utils.data.DataLoader(dataset=Testdata, batch_size=config.batch_size,
                                                      shuffle=False, drop_last=False)
    # loss function
    criterion = Myloss.Smooth_L1_XYZ_loss()

    # start training
    start_full_time = time.time()
    display_interval = 50
    total_error = {'loss': 0, 'MAE': 0, 'MAE_XYZ': 0, 'REL': 0, 'BAD1SCORE': 0, 'BAD2SCORE': 0, 'BAD5SCORE': 0, 'BAD10SCORE': 0}
    total_count = 0
    mean_error = 0
    local_step = 0
    tq = tqdm.tqdm(total=len(TestImgLoader) * config.batch_size, dynamic_ncols=True)
    tq.set_description('Test model')
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            error, count, depth = Test(model, criterion, sample, config)
            if count < 0:
                continue
            else:
                total_count += count
                mean_error = metrics.avg_error(total_error, error, total_count, count)

            tq.update(config.batch_size)
            tq.set_postfix({'Test_avg': mean_error['loss'], 'Test_cur': error['loss']})
            local_step += 1
            writer.add_scalar('Test/loss', error['loss'], local_step)
            writer.add_scalar('Test/MAE', error['MAE'], local_step)

            if local_step % display_interval == 0:
                display.display_color_depth(config.epoch, local_step, writer, sample['left'], depth, sample['depth'],
                                            sample['mask'], phase='Test',is_return_img=False, color_reverse=True)
        logger.info(f'***************************** epoch{config.epoch} Test has finished, the metrics as show below:')
        for item, value in mean_error.items():
            logger.info(f'##################### {item} = {value}....................')
    tq.close()

    logger.info('full testing time = %.2f HR' % ((time.time() - start_full_time) / 3600))
