import cv2, os, glob, pdb
import torch, datetime, time

import numpy as np
from models.model import E2ENet
from data.dataloader import get_train_loader

from utils.common import merge_config, get_work_dir, dist_print
from utils.common import save_model, get_logger, cp_projects
from utils.factory import get_optimizer, get_scheduler
from utils.dist_utils import dist_tqdm 
from utils.loss import Multi_Loss


def train(net, train_loader, criterion, optimizer, scheduler, logger, epoch, device):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    total_loss = 0
    for b_idx, (images, labels) in enumerate(progress_bar):
        t_data_1 = time.time()
        
        t_net_0 = time.time()
        
        preds = net(images.to(device))
        
        t_net_1 = time.time()
        
        loss = criterion(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        t_data_1 = time.time()
        
        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix(loss = '%.3f' % float(loss), data_time = '%.3f' % float(t_data_1 - t_data_0), net_time = '%.3f' % float(t_net_1 - t_net_0))
        
        total_loss += loss.item()
    
    logger.add_scalar('metric/loss', total_loss / len(train_loader), global_step=epoch)
    logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    
    train_loader = get_train_loader(cfg.batch_size, cfg.data_root, cfg.dataset, distributed)
    
    net = E2ENet(Channels = 96, nums_lane=4, culomn_channels = cfg.griding_num, row_channels = cfg.row_num, initialed = True)
    
    net.to(device)
    
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
        
        
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0
    
    
    optimizer = get_optimizer(net, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    logger = get_logger(work_dir, cfg)
    cp_projects(work_dir)
    
    criterion = Multi_Loss(lambda_1 = cfg.lambda_1, lambda_2 = cfg.lambda_2)
    
    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, criterion, optimizer, scheduler, logger, epoch, device)
        save_model(net, optimizer, epoch, work_dir, distributed)
 
    
        




