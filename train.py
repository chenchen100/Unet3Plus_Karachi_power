import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.callbacks import LossHistory
from utils.dataloader import Dataset, dataset_collate
from utils.utils import logger
from settings import *
import math
from functools import partial
import os
import torch
from nets.loss import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score
from nets.unet3Plus import UNet3Plus

def init_weights(net, init_type='normal', init_gain=0.02):
    from utils.utils import logger
    init_weights_logger = logger("init_weights")

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    init_weights_logger.log('initialize network with %s type' % init_type)
    net.apply(init_func)


def load_pretrained_weights(model_path, model):
    from utils.utils import logger
    load_pretrained_weights_logger = logger("load_pretrained_weights")
    if model_path != '':
        load_pretrained_weights_logger.log('Load pretrained weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes, \
                  fp16, scaler, save_period, save_dir, local_rank=0):
    from utils.utils import logger
    logger = logger("train")

    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        logger.log('Start Train')

        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # 计算f_score
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # 计算f_score
                    _f_score = f_score(outputs, labels)

            #   反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        logger.log('Finish Train')
        logger.log('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            #   计算f_score
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        logger.log('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        logger.log('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        logger.log('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))


if __name__ == "__main__":
    train_logger = logger("train")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # model = Unet(num_classes=num_classes, pretrained=pretrained).train()
    model = UNet3Plus(n_classes=num_classes).train()

    init_weights(model)

    load_pretrained_weights(train_model_path, model)

    loss_history = LossHistory(logs, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
        train_logger.log("use fp16")
    else:
        scaler = None

    model_train = model.train()

    if cuda:
        train_logger.log("use cuda")
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 读取数据集对应的txt
    with open(train_dataset, "r") as f:
        train_lines = f.readlines()
    with open(val_dataset, "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_logger.log("Adaptive adjustment of learning rate")
    # 判断当前batch_size，自适应调整学习率
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    }[optimizer_type]

    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epoch)

    # 判断每一个世代的长度
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_logger.log("load dataset")
    train_dataset = Dataset(train_lines, input_shape, num_classes, True, dataset)
    val_dataset = Dataset(val_lines, input_shape, num_classes, False, dataset)

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True,
                     drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

    # 开始模型训练
    for epoch in range(init_epoch, total_epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, total_epoch, cuda, dice_loss, focal_loss,
                      cls_weights, num_classes, fp16, scaler, save_period, logs, local_rank)

    if local_rank == 0:
        loss_history.writer.close()
