import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos, Gating_DataEmbedding, TokenEmbedding, TemporalEmbedding_v3, TimeFeatureEmbedding
from layers.StandardNorm import Normalize
import pytorch_wavelets as ptw
import pywt
from pytorch_wavelets import DWTForward
from layers.SelfAttention_Family import FullAttention, AttentionLayer, TSMixer, ResAttention
import math

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend
    

import math
# class MultiScaleSeasonMixing(nn.Module):
#     """
#     Bottom-up mixing season pattern
#     """

#     def __init__(self, config, wavelet='db4', levels=3, mode='zero', scale=0.2):
#         super(MultiScaleSeasonMixing, self).__init__()
#         self.wavalet = wavelet
#         self.levels = levels
#         self.mode = mode
#         # self.sacle_size = [[] for j in range(config.down_sampling_layers)]
#         if  wavelet=='db4':
#             kernel_length = 4
#         elif  wavelet=='bior3.7':
#             kernel_length = 8
#         else:
#             print('please define!!!!')
        
        
#         # 初始化小波变换对象
#         # self.dwtnet = DWTForward(wave=wavelet, J=levels, mode=mode).cuda()
#         # self.idwtnet = DWTForward(wave=wavelet, mode=mode).cuda()
#         self.dwtnet = ptw.DWT1D(wave=wavelet, J=levels, mode=mode)
#         self.idwtnet = ptw.IDWT1D(wave=wavelet, mode=mode)
#         #多尺度滤波器
#         print('------------------self.weights paramters in MultiScaleSeasonMixing---------------------')
#         self.cycle_weights = [[] for j in range(config.down_sampling_layers +1)]
#         self.last_weights = []
#         for j in range(config.down_sampling_layers +1):
#             size  = config.seq_len // (config.down_sampling_window ** j)
#             for i in range(levels):
#                 size = math.ceil(size/2 + kernel_length - 1)
#                 self.cycle_weights[j].append(nn.Parameter(scale * torch.randn(1, 1, size )))
#                 print(self.cycle_weights[j][i].shape )
#             self.last_weights.append(nn.Parameter(scale * torch.randn(1, 1, size )))
        
#     def wavenetdomp(self, x, weights, ca_weight):
#         """
#         将时序数据分解为趋势项和周期项
#         :param x: 输入张量，形状为 (b, t, c)
#         :return: (trend, cycle) 形状均为 (b, t, c)
#         """
#         b, c, t = x.shape

#         # 多尺度分解 
#         coeffs = self.dwtnet(x)  # 返回元组: (cA_J, [cD1, cD2, ..., cD_J])
#         cA_last = coeffs[0]  # 最底层近似系数 (b, c, t/2^J)
#         cD_list = coeffs[1]  # 各层细节系数列表
 
#         # 滤波重构周期分量
#         cycle_coeffs = []
#         for j in range(len(cD_list)):
#             cycle_coeff = cD_list[j] * weights[j].to(x.device)
#             cycle_coeffs.append(cycle_coeff)    
            
#         ca_last = cA_last * ca_weight.to(x.device)
#         # cycle_coeffs = (torch.zeros_like(cA_last), cycle_coeffs)        
#         cycle_coeffs = (ca_last, cycle_coeffs)        
        
#         #转回时域
#         cycle = self.idwt(cycle_coeffs)        

#         return  cycle
    
#     def idwt(self, x):
#         return self.idwtnet(x)
            
#     def forward(self, season_list):
#         # mixing high->low
#         out_season_list = []

#         for i in range(len(season_list)):
#             out_ = season_list[i]            
#             out_ = self.wavenetdomp(out_, self.cycle_weights[i], self.last_weights[i])   
#             # print('season:',i, out_.shape)
#             out_season_list.append(out_.float().permute(0, 2, 1))

#         return out_season_list

class WaveletDecomp(nn.Module):
    def __init__(self,configs, wavelet='db4', mode='zero', scale=0.2):
        super(WaveletDecomp, self).__init__()
        """
        将时序数据分解为高频项和低频项
        :param wavelet: 小波基类型，如 'db4', 'bior3.7' 
        :param levels: 分解层数（建议 <5）
        :param mode: 边界处理模式 ('symmetric', 'zero', 'periodic')
        """
        self.wavalet = wavelet
        self.levels = configs.levels
        self.mode = mode
        # define scale_size
        self.scale_size = []
        if  wavelet=='db4':
            kernel_length = 4
        elif  wavelet=='bior3.7':
            kernel_length = 8
        else:
            print('please define!!!!')
        size  = configs.seq_len
        total_size = 0
        for i in range(self.levels):
            size = math.ceil(size/2 + kernel_length - 1)
            self.scale_size.append(size)
            total_size+=size
        total_size+=size
        # 初始化小波变换对象
        self.dwtnet = ptw.DWT1D(wave=wavelet, J=self.levels, mode=mode)
        self.idwtnet = ptw.IDWT1D(wave=wavelet, mode=mode)
        # 多尺度滤波器
        self.right_weights = [
                nn.Parameter(torch.randn(1, 1, self.scale_size[i] )) 
                for i in range(self.levels)
            ]          
        self.left_weights = nn.Parameter( torch.randn(1, 1, self.scale_size[-1] )) 

        
    def forward(self, x, is_filted=True):
        """
        将时序数据分解为趋势项和周期项
        :param x: 输入张量，形状为 (b, t, c)
        :return: (high, low) 形状均为 (b, scale, c)
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # 转为 (b, c, t) 适应DWT输入

        # 多尺度分解 
        coeffs = self.dwtnet(x)  # 返回元组: (cA_J, [cD1, cD2, ..., cD_J])
        cA_last = coeffs[0]  # 最底层近似系数 (b, c, t/2^J)
        cD_list = coeffs[1]  # 各层细节系数列表
        # 滤波重构趋势分量
        # zeros_list = [torch.zeros_like(cD) for cD in cD_list] #仅保留趋势分量
        if is_filted:
            # low_coeffs = (cA_last * self.left_weights, zeros_list)
            low_coeffs = cA_last * self.left_weights.to(x.device)
        else:
            # low_coeffs = (cA_last, zeros_list)
            low_coeffs = cA_last
        low_coeffs = low_coeffs.permute(0, 2, 1)

        # 滤波重构周期分量
        high_coeffs = []
        for j in range(len(cD_list)):
            if is_filted:
                cycle_coeff = cD_list[j] * self.right_weights[j].to(x.device)
            else:
                cycle_coeff = cD_list[j]
            high_coeffs.append(cycle_coeff.permute(0, 2, 1))    
        high_coeffs = torch.concat(high_coeffs, dim=1)

        all_coeffs = torch.concat([low_coeffs, high_coeffs], dim=1)
        return all_coeffs

    def idwt(self, x):
        return self.idwtnet(x)



class MultiScaleSeasonMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.MultiScale_Season_TimeFreq_Att = torch.nn.ModuleList([
                        AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False)
                        , configs.d_model
                        , configs.n_heads)                        
                        for _ in range(configs.down_sampling_layers+1)
                ])
        self.frequency_domin_decomp = WaveletDecomp(configs)

    def forward(self, season_list):

        out_season_list = []
        l=0
        all_freqs = self.frequency_domin_decomp(season_list[0], is_filted=True)
        for season in season_list:
            season_res, _ = self.MultiScale_Season_TimeFreq_Att[l](season, all_freqs, all_freqs, None)
            season_res = season_res + season
            out_season_list.append(season_res)
            l+=1
        return out_season_list #{[B, L, D], [B, L//2, D], [B, L//4, D]}

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.MultiScale_Trend_TimeFreq_Att = torch.nn.ModuleList([
                        AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False)
                        , configs.d_model
                        , configs.n_heads)                        
                        for _ in range(configs.down_sampling_layers+1)
                ])
        self.frequency_domin_decomp = WaveletDecomp(configs)

    def forward(self, trend_list):

        out_trend_list = []
        l=0
        # all_freqs = self.frequency_domin_decomp(trend_list[0], is_filted=True)
        for trend in trend_list:
            # print('trend: ', l, trend.shape, all_freqs.shape)
            all_freqs = self.frequency_domin_decomp(trend, is_filted=False)
            trend_res, _ = self.MultiScale_Trend_TimeFreq_Att[l](trend, all_freqs, all_freqs, None)
            # t = trend.shape[1]
            # # trend_res, _ = self.MultiScale_Trend_TimeFreq_Att[l](trend, all_freqs[:, :t, :], trend, None)
            # trend_res, _ = self.MultiScale_Trend_TimeFreq_Att[l](all_freqs[:, :t, :], trend, trend, None)
            trend_res = trend_res + trend
            out_trend_list.append(trend_res)
            l+=1
        return out_trend_list #{[B, L, D], [B, L//2, D], [B, L//4, D]}


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            if configs.is_dff:
                self.cross_layer = nn.Sequential(
                    nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                    nn.GELU(),
                    nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
                )
            else:
                self.cross_layer = nn.Linear(in_features=configs.d_model, out_features=configs.d_model)
           
        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        if configs.is_dff:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        else:
            self.out_cross_layer = nn.Linear(in_features=configs.d_model, out_features=configs.d_model)
            

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            # print(out_season.shape, out_trend.shape)
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs, use_future_temporal_feature=True):
        super(Model, self).__init__()
        if configs.is_seed:
            torch.manual_seed(configs.seed)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.patch_window = configs.down_sampling_window
        print('configs.e_layers:',configs.e_layers)
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = Gating_DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            num =1
        else:
            self.enc_embedding = Gating_DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            num =configs.enc_in

        self.patch_size=[num]
        for i in range(self.configs.down_sampling_layers):
            self.patch_size.append(self.patch_window*(i+1) * num)
        self.TokenEmbeddings = torch.nn.ModuleList([   
                        TokenEmbedding(patch_in, configs.d_model)
                        for patch_in in self.patch_size
                    ])

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        # num_hid_seq = 0
        # for i in range(configs.down_sampling_layers + 1):
        #     num_hid_seq += configs.seq_len // (configs.down_sampling_window ** i)
        # self.aug_predictor = torch.nn.Linear( num_hid_seq, configs.seq_len)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        # configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // patch_in,
                        configs.pred_len,
                    )
                    # for i in range(configs.down_sampling_layers + 1)
                    for patch_in in self.patch_size
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
            
        # self.out_predictor = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        # flg=0
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                # print('flag:',flg,x.shape, x_1.shape, x_2.shape)
                # flg+=1
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_patchs_process_inputs(self, x_enc, x_mark_enc):
        # print(x_enc.shape, x_mark_enc.shape)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        B, T, C  = x_enc.shape

        x_enc_sampling_list = [] #创建多尺度list
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.view(B, T, 1, C)) #加入原始粒度的尺度数据
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers): #加入粗粒度的尺度数据
            patch_len = self.patch_window*(i+1)
            patch_num = T//patch_len
            # patch
            x_enc_sampling = x_enc_ori.view(B, patch_num, patch_len, C)
            # print('down sample:' , x_enc_ori.shape, x_enc_sampling.shape)
            # x_enc_sampling = self.patch_embedding[i](x_enc_sampling)
            x_enc_sampling_list.append(x_enc_sampling)

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::patch_len, :])
                # x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.patch_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, is_aug):

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        # print('before mul:',x_enc.shape, x_mark_enc.shape)
        x_enc, x_mark_enc = self.__multi_patchs_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                # print('atfer mul:',x.shape, x_mark.shape)
                B, K, P, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 3, 1, 2).contiguous().reshape(B * N, K, P)
                    x_mark = x_mark.repeat(N, 1, 1)
                # print(i, x.shape, x_mark.shape)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        # print('#'*100)
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                # print(i, 'before embed',x.shape, x_mark.shape)
                x = self.TokenEmbeddings[i](x)
                x_mark = self.enc_embedding(None, x_mark)  # [B,T,C]
                # print(i, 'after embed',x.shape, x_mark.shape)
                enc_out = x + x_mark
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                # enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out = self.TokenEmbeddings[i](x)
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # print('enc_out_list shape:',len(enc_out_list),enc_out_list[0].shape, enc_out_list[1].shape, enc_out_list[-1].shape)
        if is_aug:
            dec_out_list, dec_out_list1 = self.future_multi_mixing(B, enc_out_list, x_list)
            
            # dec_out1 = torch.concat(dec_out_list1, dim=1)
            dec_out1 = torch.stack(dec_out_list1, dim=-1)
            dec_out1 = dec_out1.sum(-1)
            
            dec_out = torch.stack(dec_out_list, dim=-1)
            dec_out = dec_out.sum(-1)
             
            # dec_out1 = self.normalize_layers[0](dec_out1, 'denorm')            
            dec_out = self.normalize_layers[0](dec_out.view(B, self.pred_len, 1, self.configs.c_out), 'denorm')            
            # print('out: ', dec_out.shape, dec_out1.shape)
            dec_out = dec_out.view(B, self.pred_len, self.configs.c_out)
            return dec_out1, dec_out 
        else:
            dec_out_list, _ = self.future_multi_mixing(B, enc_out_list, x_list)

            dec_out = torch.stack(dec_out_list, dim=-1)

            dec_out = dec_out.sum(-1)
            
            # dec_out = self.out_predictor(torch.concat([dec_out, dec_out], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            # dec_out = self.normalize_layers[0](dec_out, 'denorm')
            dec_out = self.normalize_layers[0](dec_out.view(B, self.pred_len, 1, self.configs.c_out), 'denorm')            
            # print('out: ', dec_out.shape, dec_out1.shape)
            dec_out = dec_out.view(B, self.pred_len, self.configs.c_out)
            return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        dec_out_list1 = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1))  # align temporal dimension
                # dec_out_list1.append(dec_out.reshape(B, -1, self.pred_len).permute(0 , 2, 1))
                dec_out = dec_out.permute(0 , 2, 1)
                dec_out_list1.append(dec_out.reshape(B, self.pred_len,-1))
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                # print('dec_out: ', i, dec_out.shape)
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1))# align temporal dimension
                # dec_out_list1.append(dec_out.reshape(B, -1, self.pred_len).permute(0 , 2, 1))
                dec_out = dec_out.permute(0 , 2, 1)
                dec_out_list1.append(dec_out.reshape(B, self.pred_len,-1))
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list, dec_out_list1

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, is_aug=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, is_aug)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
