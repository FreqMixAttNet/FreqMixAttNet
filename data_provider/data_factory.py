from data_provider.data_loader import Dataset_day,Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_PEMS, Dataset_Solar
from data_provider.uea import collate_fn
from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np

data_dict = {
    'Dataset_day': Dataset_day,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
}
user_ivr= True

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __len__(self):
        return sum(len(d) for d in self.datasets)
    
    def inverse_transform(self, data):
        return self.datasets.inverse_transform(data)
    
    def __getitem__(self, idx):
        dataset_idx = 0
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
            dataset_idx += 1
        raise IndexError("Index out of range")
        

def data_provider(args, flag, user_ivr=False, user_xma=False):
    # seed = 2025
    g = torch.Generator()
    if args.is_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark=False
        # torch.backends.cuda.enable_flash_sdp(True)
        g.manual_seed(args.seed)
        np.random.seed(args.seed)

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'pred') else True
    drop_last = False # if (flag == 'test' or flag == 'pred') else True
    batch_size = args.batch_size
    freq = args.freq

    
    
    if args.task_name == 'long_term_forecast' and args.target == 'ivr_rengong_session_cnt':
        drop_last = False
        if flag == 'pred':
            data_set = Data(
                    args = args,
                    data_path = args.train_path, 
                    flag = flag,
                    target=args.target,
                    features=args.features
                )
        else:
            data_set = Data(
                    args = args,
                    data_path = args.train_path, 
                    flag = flag,
                    target=args.target,
                    features=args.features
                )
            # if data_set is not None:
            #     # print(args.train_path, args.test_path, flag, len(data_set))
            # else:
            #     print(flag, 'the dataset is no need to build!')
        data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                generator=g,
                pin_memory=True
            )
        return data_set, data_loader    
    
    if 'forecast' in args.task_name:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        # if data_set is not None:
        #     # print(args.data_path, flag, len(data_set))
        # else:
        #     print(flag, 'the dataset is no need to build!')
        if args.is_seed:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                generator=g
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last            )
        # print('rng_state:',torch.get_rng_state())

        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        
        if args.is_seed:
            data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            generator=g
            )
        else:
            data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
