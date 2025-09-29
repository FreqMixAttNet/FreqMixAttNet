import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_wavelets as ptw
import pywt
from pytorch_wavelets import DWTForward

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class wavelet_series_decomp(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, config, wavelet='bior3.7', levels=4, mode='zero', scale=1):
        super(wavelet_series_decomp, self).__init__()
        self.wavalet = wavelet
        self.levels = levels
        self.mode = mode
        # self.sacle_size = [[] for j in range(config.down_sampling_layers)]
        if  wavelet=='db4':
            kernel_length = 4
        elif  wavelet=='bior3.7':
            kernel_length = 8
        else:
            print('please define!!!!')
        
        down_sampling_layers = 0
        seq_len  =  128
        # 初始化小波变换对象
        # self.dwtnet = DWTForward(wave=wavelet, J=levels, mode=mode).cuda()
        # self.idwtnet = DWTForward(wave=wavelet, mode=mode).cuda()
        self.dwtnet = ptw.DWT1D(wave=wavelet, J=levels, mode=mode)
        self.idwtnet = ptw.IDWT1D(wave=wavelet, mode=mode)
        #多尺度滤波器
        print('------------------self.weights paramters in MultiScaleSeasonMixing11---------------------')
        # self.cycle_weights = [[] for j in range(down_sampling_layers +1)]
        # self.last_weights = []
        # for j in range(down_sampling_layers +1):
        #     size  = seq_len // (down_sampling_window ** j)
        #     for i in range(levels):
        #         size = math.ceil(size/2 + kernel_length - 1)
        #         self.cycle_weights[j].append(nn.Parameter(scale * torch.randn(1, 1, size )))
        #         print(self.cycle_weights[j][i].shape )
        #     self.last_weights.append(nn.Parameter(scale * torch.randn(1, 1, size )))
        
    def wavenetdomp(self, x):
        """
        将时序数据分解为趋势项和周期项
        :param x: 输入张量，形状为 (b, t, c)
        :return: (trend, cycle) 形状均为 (b, t, c)
        """
        b, c, t = x.shape

        # 多尺度分解 
        coeffs = self.dwtnet(x)  # 返回元组: (cA_J, [cD1, cD2, ..., cD_J])
        cA_last = coeffs[0]  # 最底层近似系数 (b, c, t/2^J)
        cD_list = coeffs[1]  # 各层细节系数列表
 
        # 滤波重构周期分量
        cycle_coeffs = []
        # print("cA_last:",cA_last.shape,x.shape)
        for j in range(len(cD_list)):
            # cycle_coeff = cD_list[j] .to(x.device)

            cycle_coeff = torch.zeros_like(cD_list[j].to(x.device) )
            cycle_coeffs.append(cycle_coeff)    
            
        ca_last = cA_last .to(x.device)
        # cycle_coeffs = (torch.zeros_like(cA_last), cycle_coeffs)        
        cycle_coeffs = (ca_last, cycle_coeffs)        
        
        #转回时域
        cycle = self.idwt(cycle_coeffs)        

        return  cycle.permute(0, 2, 1)
    
    def idwt(self, x):
        return self.idwtnet(x)
            
    def forward(self, season_list):
        # mixing high->low
      
        out_ = self.wavenetdomp(season_list)   
        out_ = out_.float()

        return out_

class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
