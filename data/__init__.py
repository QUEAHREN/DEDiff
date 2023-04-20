'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data

# def create_dataloader(dataset, dataset_opt, phase, opt=None, sampler=None):
#     if phase == 'train':
#         if opt['dist']:
#             world_size = torch.distributed.get_world_size()
#             num_workers = dataset_opt['n_workers']
#             assert dataset_opt['batch_size'] % world_size == 0
#             batch_size = dataset_opt['batch_size'] // world_size
#             shuffle = False
#         else:
#             num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
#             batch_size = dataset_opt['batch_size']
#             shuffle = True
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                                            num_workers=num_workers, sampler=sampler, drop_last=True,
#                                            pin_memory=False)
#     else:
#         return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
#                                            pin_memory=False)

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode == 'HR':
        from data.LRHR_dataset import LRHRDataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR')
                    )
    elif mode == 'LQGT_event':
        from data.LQGT_dataset_mat import LQGTDataset_mat as D  
        dataset = D(dataset_opt)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
