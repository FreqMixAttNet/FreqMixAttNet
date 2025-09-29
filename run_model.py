import argparse
import torch
import pandas as pd
# from model import DlinearModel
from data_provider.data_loader import Dataset_day
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.xma_ivr_exp_long_term_forecasting import Fusion_Exp_Long_Term_Forecast
import torch.backends
import time, random
import numpy as np
# from cal_metrics import cal_mape_ratio

import os
# 仅设置一块可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 设置多块可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if_fusion = False
file2 = "record_results.txt"

type = 'IVR_RENGONG'

# sbatch ./huawu-weili.slurm
# scancel -v your_jobid


parser = argparse.ArgumentParser(description="Process pySpark arguments.")
# 添加参数
# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='09_', help='model id')
parser.add_argument('--model', type=str, default='TimeMixerFilter3_5', help='model name, options: [ Dlinear,PatchTST,Pyraformer,SparseTSF,iTransformer,TimeMixer,TimeMixer2, TimeMixer3, PaiFilter, CycleNet, TimeMixerFilter3_1,MaskTimeMixerFilter, WaveFilterMixNet0_1')

# /root/task
 # data loader
parser.add_argument('--data', type=str,default="ETTh1", help='dataset type')
parser.add_argument('--if_vaild', type=bool, default=True, help='if use valid data or not~')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default= 'ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--scaler_type', type=str, default='std', help='[std, minmax, RevIN]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument("--start_test_days", type=str, default="2025-03-15")
parser.add_argument("--start_pred_days", type=str, default="2025-06-06")
parser.add_argument("--pred_period", type=int, default=192) # 预估 47 天数据
parser.add_argument("--label_name", type=str, default="OT")
parser.add_argument("--seq_len", type=int, default=96)
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument("--label_len", type=int, default=0)
parser.add_argument("--num_class", type=int, default=1)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')



# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeH', help='time features encoding, options:[timeF, fixed, learned, timeH]')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')



parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='wavelet',
                    help='method of series decompsition, support wavelet or moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=24, help='the length of segmen-wise iteration of SegRNN')
parser.add_argument('--is_dff', type=bool, default=False, help='use augmentation')

# TimeXer
parser.add_argument('--patch_len', type=int, default=16, help='patch length')

# wavenet decomp:levels
parser.add_argument('--levels', type=int, default=3, help='levels decomp of wavenet')


# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='train', help='exp description')
parser.add_argument('--loss', type=str, default='MSE_penalty2', help='loss function, option:[MSE, MSE_penalty, MSE_penalty2]')
parser.add_argument('--mape_threshold', type=float, default=0.12, help='the mape threshold for select elements.')
parser.add_argument('--penalty_threshold', type=float, default=0.8, help='the mape ratio threshold for select time series list')
parser.add_argument('--penalty_weight', type=float, default=0, help='the weight of loss penalty')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--seed', type=int, default=42, help="Randomization seed")
parser.add_argument('--is_seed', type=bool, default=True, help='use seed')

# GPU SETTING
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
# parser.add_argument('--gpu', type=int, default=4, help='gpu')
# parser.add_argument('--gpu', type=int, default=1 if torch.cuda.is_available() else None)

parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation SETTING
parser.add_argument('--use_augmentation', type=bool, default=True, help='use augmentation')
parser.add_argument('--aug_weight', type=float, default=0.07, help='the weight of loss for augmentation')
parser.add_argument('--jitter_scale_ratio', type=float, default=0.01, help='jitter_scale_ratio for weak augmentation')
parser.add_argument('--jitter_ratio', type=float, default=0.3, help='jitter_ratio of type1 for strong augmentation')
parser.add_argument('--max_seg', type=int, default=12, help='max_seg of type1 for strong augmentation')
parser.add_argument('--aug_type', type=str, default='type2', help='type for strong augmentation')
parser.add_argument('--mix_rate', type=float, default=0.8, help='freq mix rate of type2 for strong augmentation')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='freq mix rate of type2 for strong augmentation')

parser.add_argument('--period_len', type=int, default=8, help='period_len for model spareseTSF')

parser.add_argument('--freq_loss', type=bool, default=True, help='use augmentation')
parser.add_argument('--freq_weight', type=float, default=4.0, help='use augmentation')

parser.add_argument('--use_swa', type=bool, default=False, help='use swa for update lr and params')
parser.add_argument('--ver', type=str, default='type2', help='type for strong augmentation')
parser.add_argument('--trend_weight', type=float, default=0.5, help='trend_weight')
parser.add_argument('--l1l2_alpha', type=float, default=0.035, help='l1l2_alpha')
parser.add_argument('--aug_constrast_weight1', type=float, default=0.01, help='aug_constrast_weight1')
parser.add_argument('--aug_constrast_weight2', type=float, default=0.008, help='aug_constrast_weight2')

args, unknown = parser.parse_known_args()


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)


dt = args.start_test_days
pred_period = args.pred_period

if args.features == 'S':
    args.enc_in=1
    args.dec_in=1
    args.c_out=1

    
    
date_beg = args.start_test_days
days = 31
length = args.pred_len
type = 'ETTh1'
model_type = args.model

from exp.exp_long_term_forecasting_Contr3 import Exp_Long_Term_Forecast
    
Exp = Exp_Long_Term_Forecast
    
import pandas as pd
results_list = []
flg = 0
index = 0
if args.is_training:
    for ii in range(0,args.itr):   
        time_now = time.time()
        index +=1
        exp = Exp(args)  # set experiments
        # version = "iter:" + str(index)+ "   "+ "learning_rate:" + str(args.learning_rate)+" "  + "dropout:" + str(args.dropout) 
        version = "iter:" + str(index)+ "   "+ "alpha:" + str(args.alpha)+" "  + "aug_weight:" + str(args.aug_weight) 

        setting = 'ver_{}_us_{}_lr{}_dr{}_{}_sl{}_pl{}_dm{}_nh{}_aw{}_alpha{}_fw{}_ag1c{}_ag2c{}_bs{}_{}_{}_{}_ft{}_el{}_dl{}_df{}_id{}_eb{}_dsl{}_dsw{}_ls{}_pw{}_{}_{}'.format(
            args.ver,
            args.use_swa,
            args.learning_rate,
            args.dropout,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.aug_weight,
            args.alpha,
            args.freq_weight,
            args.aug_constrast_weight1,
            args.aug_constrast_weight2,                        
            args.batch_size,                         
            args.task_name,
            args.model_id,
            "fma",                       
            args.features,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.is_dff,
            args.embed,
            args.down_sampling_layers,
            args.down_sampling_window,
            args.loss,
            args.penalty_weight,
            
            args.des, ii)

        print("*************************************************************************************************** *************************")
        print(version)
        print("*************************************************************************************************** *************************")
        print("#" * 80)
        print('>>>>>>>start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        current_time, rmse, mae, mape, mspe, mse, total_params = exp.test(setting)
        print('#'*80)
        print(f'\nTotal spend time for itr {flg}: ', time.time() - time_now)
        print('\n')
        print('#'*80)

        f = open(file2, 'a')
        # f.write(str(args.learning_rate)+'|  ' + setting + "  | \n")
        f.write(version + "\n")  # 保留列名，忽略索引
        f.write(setting + '\n')
        
        f.write('{} | rmse:{}, mae:{}, mape:{}, mspe{}, mse{}'.format(current_time, rmse, mae, mape, mspe, mse))
        f.write('\n')
        f.write('\n')
        f.close()
        
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
        flg+=1

        iteration = "iter:" + str(index)
        # results_list.append([current_time, args.learning_rate, args.model, rmse, mae, mape, mspe, mse, total_params])
        results_list.append([iteration,version, mae,  mse])
        # results_list.append([current_time, args.learning_rate, args.model, rmse, mae, mape, mspe, mse, total_params])
        
else:
    exp = Exp(args)  # set experiments
    ii = 0
    setting = 'ver_{}_us_{}_lr{}_dr{}_{}_{}_{}_{}_ft{}_sl{}_prl{}_pl{}_dm{}_bs{}_nh{}_el{}_dl{}_df{}_id{}_ma{}_st{}_eb{}_dsl{}_dsw{}_ls{}_mt{}_pt{}_pw{}_ua{}_ut{}_aw{}_jr{}_fw{}_agc1{}_agc2{}_{}'.format(
                args.ver,
                args.use_swa,
                args.learning_rate,
                args.dropout,
                args.task_name,
                args.model_id,
                "fma",
                args.data,
                args.features,
                args.seq_len,
                args.period_len,
                args.pred_len,
                args.d_model,
                args.batch_size,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.is_dff,
                args.moving_avg,
                args.scaler_type,
                args.embed,
                args.down_sampling_layers,
                args.down_sampling_window,
                args.loss,
                args.mape_threshold,
                args.penalty_threshold,
                args.penalty_weight,
                args.use_augmentation,
                args.aug_type, 
                args.aug_weight,
                args.freq_weight,
                args.aug_constrast_weight1,
                args.aug_constrast_weight2,
                args.des, ii)

    print('>>>>>>>testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    current_time, rmse, mae, mape, mspe, mse = exp.test(setting, test='pred')
    
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()
results_list = pd.DataFrame(results_list, columns= [ 'iteration','version', 'mae' , 'mse'])
print(results_list)

