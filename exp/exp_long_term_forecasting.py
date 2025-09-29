from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, MAPE,MSE, MAE
import torch
import torch.nn as nn
from torch import optim
import os
import time, random
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single,DataTransform ,DataTransform_FD, scaling, freq_mix
from utils.losses import *
from layers.Autoformer_EncDec import series_decomp,wavelet_series_decomp

import torch.nn.functional as F
import torch.fft as fft
warnings.filterwarnings('ignore')


class mape_loss_with_penalty2(nn.Module):
    def __init__(self, mape_threshold=0.15, penalty_threshold=0.7, penalty_weight=5.0,alpha=0.1):
        """
        初始化带有惩罚项的均方误差损失函数。        
        参数:
        - mape_threshold: MAPE筛选阈值，默认为0.15
        - penalty_threshold: 样本惩罚阈值，默认为0.7
        - penalty_weight: 惩罚项的权重，默认为1.0
        """
        super(mape_loss_with_penalty2, self).__init__()
        self.mape_threshold = mape_threshold
        self.penalty_threshold = penalty_threshold
        self.penalty_weight = 0
        self.combie_loss = True
        self.alpha = alpha
    def forward(self, pred, target):
        """
        计算损失值。        
        参数:
        - preds: 预测张量 [B, T, 1]
        - targets: 真实值张量 [B, T, 1]        
        返回:
        - total_loss: 结合了MSE和惩罚项的总损失
        """
        assert pred.shape == target.shape, "预测和真实值形状需一致" 
        preds = pred
        targets = target
        # print(pred[:10,0,:], target[:10,0,:])
        mse = torch.mean((preds - targets) ** 2)        
        epsilon = 1e-6
        absolute_percent_error = torch.abs((preds - targets) / (targets + epsilon))

        mask_small_error = (absolute_percent_error < self.mape_threshold).float()
        sample_percent = mask_small_error.mean(dim=(1,2))

        penalty_mask = (sample_percent < self.penalty_threshold).float()      
        # print('%'*50, preds.shape, targets.shape, mse, penalty_mask.shape, torch.mean((preds - targets) ** 2, dim=(1,2)).shape)
        penalized_mse = torch.mean(penalty_mask * torch.mean((preds - targets) ** 2, dim=(1,2)))        
        total_loss = mse + self.penalty_weight * penalized_mse
        # print("self.penalty_weight:",self.penalty_weight)
        #最终返回标量损失
        if(self.combie_loss):
            # print("self.penalty_weight1")

            mse = nn.MSELoss(reduction='none')(preds, targets)
            mae = nn.L1Loss(reduction='none')(preds,targets)
            # self.alpha*mse +
            # alpha = 0.035
            loss =  (1 - self.alpha)*mae
            return loss.mean()
        else:
            # print("self.penalty_weight2")
            return total_loss

    
def filtered_mape_loss(outputs, outputs1, threshold=0.1):
    """
    计算 MAPE 并返回 MAPE ≥ threshold 的样本损失
    参数:
        outputs  : 预测值，形状 [B, T, 1]
        outputs1 : 真实值，形状 [B, T, 1]
        threshold: 损失阈值（50% 对应 0.5）
    返回:
        loss_tensor: 形状 [B, 1]，仅保留 MAPE ≥ threshold 的样本损失
    """
    # 1. 计算绝对百分比误差 (APE)
    eps = 1e-8  # 避免除零
    # mape = torch.abs((outputs1 - outputs) / (outputs1 + eps))  # [B, T, 1]
    mape = (outputs - outputs1) ** 2
    # 2. 计算每个样本的平均 MAPE (沿时间步 T 平均)
    # mape = torch.mean(mape, dim=1) * 100  # [B, 1]
    
    # 3. 筛选 MAPE ≥ 50% 的样本
    mask = (mape >= threshold)  # 阈值转换为百分比
    loss_tensor = torch.where(mask, mape, torch.zeros_like(mape))
    
    return loss_tensor.mean()
    
def init_weights(m):
    """自定义初始化函数"""
    if isinstance(m, nn.Linear):
        # 全连接层：He初始化
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # 卷积层：Xavier初始化
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.LSTM):
        # LSTM层：正交初始化
        for name, param in m.named_parameters():
            if 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)


                
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        print('Exp_Long_Term_Forecast 11:',args.ver)
        if args.is_seed:
            torch.manual_seed(42)       # CPU随机种子
            torch.cuda.manual_seed_all(42)  # GPU随机种子
            torch.backends.cudnn.deterministic = True  # 消除CUDA随机性
        self.decompsition = wavelet_series_decomp(self.args.levels)
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        #对整个网络进行权重初始化
        # model.apply(init_weights)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        print("-"*40, " start to print model parameters ", "-"*40)
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: {param.shape}")
        
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {self.total_params}")
        print("-"*40, " endding of print model parameters ", "-"*40)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # self.aug_loss = mape_loss_with_penalty(self.args.mape_threshold, self.args.penalty_weight)
        self.aug_loss = hierarchical_contrastive_loss
        # self.aug_loss = nn.MSELoss()
        # self.aug_loss = hierarchical_contrastive_loss1
        if self.args.loss=='MSE_penalty':
            criterion = mape_loss_with_penalty(self.args.mape_threshold, self.args.penalty_weight)
        elif self.args.loss=='MSE_penalty2':
            criterion = mape_loss_with_penalty2(self.args.mape_threshold, self.args.penalty_threshold, self.args.penalty_weight, self.args.l1l2_alpha)
        elif self.args.loss == 'SMAPE':
            return smape_loss()
        else:
            criterion = nn.MSELoss()
        return criterion
 
    def _get_aug_output(self,batch_x,batch_x_mark,dec_inp,batch_y_mark):
        if self.args.aug_type=='type1':
            _, strong_x = DataTransform(batch_x.cpu().numpy(), self.args)
            strong_x = torch.from_numpy(strong_x).float().to(self.device)
        elif self.args.aug_type == 'type2':
            x_data_f = fft.fft(batch_x).abs()
            strong_x1 = DataTransform_FD(x_data_f, self.args)
            # print("type2:1111111" )
            _, strong_x0 = DataTransform(batch_x.cpu().numpy(), self.args)
            strong_x0 = torch.from_numpy(strong_x0).float().to(self.device)     
            strong_x11 = freq_mix(batch_x, self.args.mix_rate)
        else:
            print('pls define your target augment type!')


        strong_x = torch.cat([strong_x0,strong_x1,strong_x11, batch_x], dim=0) 
        batch_x_mark1 = torch.cat([batch_x_mark,batch_x_mark, batch_x_mark,batch_x_mark], dim=0) 
        dec_inp1 = torch.cat([dec_inp,dec_inp, dec_inp,dec_inp], dim=0) 
        batch_y_mark1 = torch.cat([batch_y_mark,batch_y_mark, batch_y_mark,batch_y_mark], dim=0) 
        # time_1 = time.time()
        strong_outputs, outputs= self.model(strong_x, batch_x_mark1, dec_inp1, batch_y_mark1, is_aug=True)
        # time_2 = time.time() - time_1
        # print('model cal cost time: ', time_2)

        # strong_x = torch.cat([strong_x, batch_x], dim=0) 
        # batch_x_mark1 = torch.cat([batch_x_mark, batch_x_mark], dim=0) 
        # dec_inp1 = torch.cat([dec_inp, dec_inp], dim=0) 
        # batch_y_mark1 = torch.cat([batch_y_mark, batch_y_mark], dim=0) 
        
        # strong_outputs, outputs,outputs_trend,outputs_seasion = self.model(strong_x, batch_x_mark1, dec_inp1, batch_y_mark1, is_aug=True)
        # print('strong_outputs, outputs: ', strong_outputs.shape, outputs.shape)
        # print('weak_outputs, strong_outputs: ', outputs.shape,outputs_trend.shape)
        # time_1 = time.time()
        _,_,_,outputs = torch.split(outputs, split_size_or_sections=len(strong_outputs)//4, dim=0)
        # _,_,outputs_trend = torch.split(outputs_trend, split_size_or_sections=len(outputs_trend)//3, dim=0)
        # _,_,outputs_seasion = torch.split(outputs_seasion, split_size_or_sections=len(outputs_seasion)//3, dim=0)
        weak_outputs0,weak_outputs1,weak_outputs11, strong_outputs = torch.split(strong_outputs, split_size_or_sections=len(strong_outputs)//4, dim=0)
        # time_2 = time.time() - time_1
        # print('split cost time: ', time_2)

        # if i==0: print('weak_outputs, strong_outputs: ', weak_outputs.shape, strong_outputs.shape)
        # alpha_shiyu = 0.7
        # loss_aug = self.aug_loss(weak_outputs0, strong_outputs,alpha_shiyu)*self.args.alpha

        # weak_outputs0:time-domain,weak_outputs1:freq-domian,weak_outputs11:mix-domain
        alpha_pingyu =  self.args.alpha
        # 0.3
        # self.args.alpha
        # time_1 = time.time()
        #
        loss_aug = self.aug_loss(weak_outputs1, strong_outputs,alpha_pingyu)
        # loss_aug = self.aug_loss(weak_outputs0, strong_outputs,alpha_pingyu)
        # loss_aug = self.aug_loss(weak_outputs11, strong_outputs,alpha_pingyu)



        # time_2 = time.time() - time_1
        # print('aug loss 1 cost time: ', time_2)
        
        # time_1 = time.time()
        # loss_aug111 = self.aug_loss(weak_outputs11,weak_outputs1,alpha_pingyu)
        # time_2 = time.time() - time_1
        # print('aug loss 2 cost time: ', time_2)

        # time_1 = time.time()
        # loss_aug110 = self.aug_loss(weak_outputs11,weak_outputs0,alpha_pingyu)
        # time_2 = time.time() - time_1
        # print('aug loss 3 cost time: ', time_2)


        # loss_aug = (torch.fft.rfft(weak_outputs1, dim=1) - torch.fft.rfft(strong_outputs, dim=1)).abs().mean() 
        # loss_c = (1 + loss_aug111 - loss_aug11_raw) + (1 + loss_aug110 - loss_aug11_raw) 
        # loss_aug += self.aug_loss(weak_outputs1, strong_outputs,self.args.alpha)

 
        # 生成随机排列
        perm_indices = torch.randperm(strong_outputs.size(0))
        # 应用排列
        a3 = strong_outputs[perm_indices]
        # print ("perm_indices:",perm_indices,strong_outputs,a3)
        # time_1 = time.time()
        loss_c = compute_triplet_loss(weak_outputs11,strong_outputs,weak_outputs1)
        # time_2 = time.time() - time_1
        # print('aug loss c cost time: ', time_2)
        # time_1 = time.time()
        loss_c1 = compute_triplet_loss(weak_outputs11,weak_outputs1,a3)
        # time_2 = time.time() - time_1
        # print('aug loss c1 cost time: ', time_2)
        return loss_aug,outputs,loss_c,loss_c1


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mape, total_mae, total_mse = [], [], []
        last30_mape = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                shape = batch_x.shape
                if len(shape)>3:
                    batch_x = batch_x.reshape(-1, self.args.seq_len, shape[-1])
                    batch_y = batch_y.reshape(-1, self.args.pred_len, shape[-1])
                    # batch_x = batch_x[:, 0, ...]
                    # batch_y = batch_y[:, 0, ...]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                      
                # encoder - decoder
                
                loss_aug,outputs,loss_aug_c,loss_aug_c1 = self._get_aug_output(batch_x,batch_x_mark,dec_inp,batch_y_mark)
                        
                # loss_aug,outputs,outputs_trend,outputs_season = self._get_aug_output(batch_x,batch_x_mark,dec_inp,batch_y_mark)

                    
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # trend_true = self.decompsition(true.permute(0, 2, 1))
                # season_true  = true-trend_true 
                # print("trand:",outputs_trend.shape, batch_y.shape,trend_pred.shape,season_pred.shape)
                loss = criterion(pred[:, -self.args.pred_len:, f_dim:], true[:, -self.args.pred_len:, f_dim:])

             
                # loss = criterion(pred, true)
                if self.args.freq_loss:
                    loss_feq = (torch.fft.rfft(pred[:, -self.args.pred_len:, f_dim:], dim=1) - torch.fft.rfft(true[:, -self.args.pred_len:, f_dim:], dim=1)).abs().mean() 
                    loss += loss_feq * self.args.freq_weight
                # aug_weight = 0.07
                aug_weight = self.args.aug_weight
                if self.args.use_augmentation:loss += aug_weight * loss_aug.detach().cpu()
                # alpha0 = 0.05
                alpha0 = self.args.aug_constrast_weight1
                loss +=  alpha0 * loss_aug_c.detach().cpu()
                alpha1= self.args.aug_constrast_weight2
                # alpha1 = 0.06
                loss += alpha1* loss_aug_c1.detach().cpu()

                # alpha0 = 0.05 alpha1=0.06
                total_loss.append(loss)
                
                shape = true.shape
                if pred.shape[-1] != true.shape[-1]:
                    pred = np.tile(pred, [1, 1, int(true.shape[-1] / outputs.shape[-1])])
                pred = pred.numpy()
                true = true.numpy()
                if self.args.inverse:
                    pred = vali_data.inverse_transform(pred.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true = vali_data.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)
                mape = MAPE(pred[:, -self.args.pred_len:, f_dim:], true[:, -self.args.pred_len:, f_dim:])
                mae = MAE(pred[:, -self.args.pred_len:, f_dim:], true[:, -self.args.pred_len:, f_dim:])
                mse = MSE(pred[:, -self.args.pred_len:, f_dim:], true[:, -self.args.pred_len:, f_dim:])
                d30_mape = MAPE(pred[:, -30:, f_dim:], true[:, -30:, f_dim:])
                total_mape.append(mape)
                total_mae.append(mae)
                total_mse.append(mse)
                last30_mape.append(d30_mape)
                
        total_loss = np.average(total_loss)
        total_mape, total_mae, total_mse = np.average(total_mape), np.average(total_mae), np.average(total_mse)
        last30_mape = np.average(last30_mape)
        self.model.train()
        return total_loss, total_mape,last30_mape, total_mae, total_mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if self.args.if_vaild:
            vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
      
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mape, total_mae, total_mse = [], [], []
            last30_mape = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                shape = batch_x.shape
                if len(shape)>3:
                    batch_x = batch_x.reshape(-1, self.args.seq_len, shape[-1])
                    batch_y = batch_y.reshape(-1, self.args.pred_len, shape[-1])
                # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        
                # encoder - decoder
                
                    # augmentation
                # loss_aug,outputs,outputs_trend,outputs_season = self._get_aug_output(batch_x,batch_x_mark,dec_inp,batch_y_mark)
                loss_aug,outputs,loss_aug_c,loss_aug_c1 = self._get_aug_output(batch_x,batch_x_mark,dec_inp,batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                true = batch_y.detach().cpu()
                # outputs_trend = outputs_trend.detach().cpu()
                # outputs_season = outputs_season.detach().cpu()
                # trend_true = self.decompsition(true.permute(0, 2, 1))
                # season_true  = true-trend_true 
                if i==0: print("trand:",outputs.shape, batch_y.shape)

                loss = criterion(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:].to(self.device))       



                if self.args.freq_loss:
                    # time_1 = time.time()
                    loss_feq = (torch.fft.rfft(outputs[:, -self.args.pred_len:, f_dim:], dim=1) - torch.fft.rfft(batch_y[:, -self.args.pred_len:, f_dim:], dim=1)).abs().mean() 

                    loss += loss_feq * self.args.freq_weight
                    # time_2 = time.time() - time_1
                    # print('fft cost time: ', time_2)
                    # print("loss_feq:",loss_feq)
                    # if i==0:  print("loss:",loss,"loss_aug:",loss_aug,"loss_aug_c:",loss_aug_c, "loss_feq",loss_feq,"loss_aug_c1:",loss_aug_c1)

                # print("loss21:",loss)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                shape = batch_y.shape
                if outputs.shape[-1] != batch_y.shape[-1]:
                    outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                
                if self.args.inverse:
                    outputs = train_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = train_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                mape = MAPE(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:])
                mae = MAE(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:])
                mse = MSE(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:])
                d30_mape = MAPE(outputs[:, -30:, f_dim:], batch_y[:, -30:, f_dim:])
                # aug_weight = 0.07
                aug_weight = self.args.aug_weight
                if self.args.use_augmentation:loss += aug_weight * loss_aug
                # self.args.alpha = 0.04
                # alpha0 = 0.05
                alpha0 = self.args.aug_constrast_weight1
                loss += alpha0 * loss_aug_c
                # alpha1 = 0.06
                alpha1 = self.args.aug_constrast_weight2
                loss += alpha1 * loss_aug_c1
                # print("loss_aug:",loss_aug)
                # print("loss33:",loss)
                train_loss.append(loss.item())
                train_mape.append(mape)
                total_mae.append(mae)
                total_mse.append(mse)
                last30_mape.append(d30_mape)
                    
                    

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self.args.use_swa:
                    self.net.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_mape, train_mse, train_mae = np.average(train_mape),np.average(total_mse),np.average(total_mae)
            train_last30_mape = np.average(last30_mape)
            if self.args.if_vaild:
                vali_loss, vali_mape, vali_last30_mape, vali_mae, vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mape, test_last30_mape, test_mae, test_mse = self.vali(test_data, test_loader, criterion)
            if self.args.if_vaild:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                print("Epoch: {0}, Steps: {1} | Train Mae: {2:.7f} Vali Mae: {3:.7f} Test Mae: {4:.7f}".format(
                    epoch + 1, train_steps, train_mae, vali_mae, test_mae))
                print("Epoch: {0}, Steps: {1} | Train Mse: {2:.7f} Vali Mse: {3:.7f} Test Mse: {4:.7f}".format(
                    epoch + 1, train_steps, train_mse, vali_mse, test_mse))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss11: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
                print("Epoch: {0}, Steps: {1} | Train Mse: {2:.7f} Test Mse: {3:.7f}".format(
                    epoch + 1, train_steps, train_mse, test_mse))
                print("Epoch: {0}, Steps: {1} | Train Mae: {2:.7f} Test Mae: {3:.7f}".format(
                    epoch + 1, train_steps, train_mae, test_mae))
                early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        if self.args.data == 'Dataset_day_Pred':
            self.test(setting, 'pred',  file="result_long_term_forecast_pred.txt")

        return self.model
    # def delete_checkpoint(setting, checkpoint_name='checkpoint.pth'):
    #     """删除指定的checkpoint文件"""
    #     checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
    #     if os.path.exists(checkpoint_path):
    #         os.remove(checkpoint_path)
    #         print(f"已删除checkpoint: {checkpoint_path}")
    #     else:
    #         print(f"Checkpoint不存在: {checkpoint_path}")
    def test(self, setting, test=0, file="result_long_term_forecast.txt"):
        if test=='pred':
            flag='pred'
        else:
            flag = 'test'
        
        test_data, test_loader = self._get_data(flag=flag)

        print('loading model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=device) )
        # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        feats = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # if i==0: print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                shape = batch_x.shape
                if len(shape)>3:
                    # batch_x = batch_x.reshape(-1, shape[-2], shape[-1])
                    # batch_y = batch_y.reshape(-1, shape[-2], shape[-1])
                    batch_x = batch_x[:, 0, ...]
                    batch_y = batch_y[:, 0, ...]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    is_plot='./plot/' + str(self.args.model)+ 'test_epoch_' + str(i)
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, is_plot=is_plot)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # if i==0:print('output:', outputs.shape, batch_x.shape, batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # shape2 = batch_x.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # batch_x = test_data.inverse_transform(batch_x.reshape(shape2[0] * shape2[1], -1)).reshape(shape2[0], shape2[1], -1)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # batch_x= batch_x[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy() if torch.is_tensor(batch_x) else batch_x
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[-1, :, -1], true[-1, :, -1]), axis=0)
                    pd = np.concatenate((input[-1, :, -1], pred[-1, :, -1]), axis=0)
                    # np.savetxt(os.path.join(folder_path, str(i) + '_trues_and_preds.npz'), gt, pd)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    feats.append(input)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        feats = np.concatenate(feats, axis=0)
        # print('test shape:', preds.shape, trues.shape, feats.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        feats = feats.reshape(-1, feats.shape[-2], feats.shape[-1])
        # print('test shape:', preds.shape, trues.shape, feats.shape)
        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
        # result save
        if test=='pred':
            folder_path = './preds/' + setting + '/'
        else:
            folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if test != 'pred':
            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1, 1)
                    y = trues[i].reshape(-1, 1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = 'Not calculated'
            from datetime import datetime
            current_time = datetime.now()
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            d30_mape = MAPE(preds[:, -30:, :], trues[:, -30:, :])
            print(current_time, 'rmse:{}, mae:{}, mape:{}, mspe{}, mse{}'.format(rmse, mae, mape, mspe, mse))
            
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'feats.npy', feats)
        checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        # if os.path.exists(checkpoint_path):
        #     os.remove(checkpoint_path)
        #     print(f"已删除checkpoint: {checkpoint_path}")
        # self.delete_checkpoint(setting)
        print('finished!')
        if test=='pred':
            return None
        else:
            return current_time, rmse, mae, mape, mspe, mse, self.total_params
