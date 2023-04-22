import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        gt_img = Metrics.tensor2img(visuals['GT'])  # uint8
        lq_img = Metrics.tensor2img(visuals['LQ'])
        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            rlt_img = visuals['rlt']  # uint8
            sample_num = rlt_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(rlt_img[iter]), '{}/{}_{}_rlt_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            rlt_img = Metrics.tensor2img(visuals['rlt'])  # uint8
            Metrics.save_img(
                rlt_img, '{}/{}_{}_rlt_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['rlt'][-1]), '{}/{}_{}_rlt.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            lq_img, '{}/{}_{}_lq.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(lq_img, Metrics.tensor2img(visuals['rlt'][-1]), gt_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
