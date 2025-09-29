# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature=0.2, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # zis, zjs: [B, T, F]
        representations = torch.cat([zjs, zis], dim=0) #[2B, T, F]

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        # print("similarity_matrix0:",x.shape,y.transpose(1,0).shape)
        # v = self._cosine_similarity(x, y.transpose(1,0))

        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        zjs = zjs.reshape(zjs.size(0), -1)  # [b, t*c]
        zis = zis.reshape(zis.size(0), -1)  # [b, t*c]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # print("similarity_matrix:",similarity_matrix.shape,representations.shape,zis.shape)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        total_size = similarity_matrix.size(0)  # 应该等于 2 * batch_size
    
        # print(f"Total size: {total_size}, batch_size: {self.batch_size}",l_pos.shape)
        if total_size != 2 * self.batch_size:
            return torch.tensor(0.0, requires_grad=True, device=similarity_matrix.device)   # 0.0
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss    
# import torch
# from torch import nn

def temp_contrastive_loss(z1, z2, alpha=0.8, temporal_unit=0, temp=1.0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    temporal_unit = 0
    # alpha = 0.9
    while z1.size(1) > 1:
        # print("Z1:",z1.shape,z2.shape)

        if d >= temporal_unit:
            # print("Z10 alpha:",alpha)

            if 1 - alpha != 0:
                if d == 0:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
                else:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
        if d>=1:
            break
            # print("loss0 :",loss)
        d += 1

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    return loss / d



def inst_contrastive_loss(z1, z2, alpha=0.8, temporal_unit=0, temp=1.0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    temporal_unit = 0
    # alpha = 0.9
    while z1.size(1) > 1:
        # print("Z1:",z1.shape,z2.shape)

        if alpha != 0:
            if d == 0:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            else:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
        if d>=1:
            break
            # print("loss0 :",loss)
        d += 1

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        # print("Z111:",z1.shape,z2.shape)

        if alpha != 0:
            loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            d += 1
            # print("loss1 :",loss)
    return loss / d




def hierarchical_contrastive_loss(z1, z2, alpha=0.8, temporal_unit=0, temp=1.0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    temporal_unit = 0
    # alpha = 0.9
    while z1.size(1) > 1:
        # print("Z1:",z1.shape,z2.shape)

        if alpha != 0:
            if d == 0:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            else:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
        # print("loss :",loss)
        if d >= temporal_unit:
            # print("Z10 alpha:",alpha)

            if 1 - alpha != 0:
                if d == 0:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
                else:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
        if d>=1:
            break
            # print("loss0 :",loss)
        d += 1

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        # print("Z111:",z1.shape,z2.shape)

        if alpha != 0:
            loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            d += 1
            # print("loss1 :",loss)
    return loss / d


def temporal_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha = 0.2
    beta = 0.2

    if T == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[:, torch.randperm(z1.shape[1]), :].view(z1.size())
    uni_z2 = beta * z2 + (1 - beta) * z2[:, torch.randperm(z1.shape[1]), :].view(z2.size())

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=1)

    sim = torch.matmul(z[:, : 2 * T, :], z.transpose(1, 2)) / temp  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    if T > 1500:
        z, sim = z.cpu(), sim.cpu()
        torch.cuda.empty_cache()

    logits = -F.log_softmax(logits, dim=-1)

    logits = logits[:, :2 * T, :(2 * T - 1)]

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def instance_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha = 0.2
    beta = 0.2

    if B == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[torch.randperm(z1.shape[0]), :, :].view(z1.size())
    uni_z2 = beta * z2 + (1 - beta) * z2[torch.randperm(z2.shape[0]), :, :].view(z2.size())

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2)) / temp  # T x 2B x 2B

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B  x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    # cur_data = z2
    # print("cur_data:",cur_data.shape)
    # ax = cur_data - torch.mean(cur_data, dim=1, keepdim=True)
    # cur_sim = torch.bmm(F.normalize(ax, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
    # print("cur_sim:",cur_sim.shape)

    logits = logits[:, :2 * B, :(2 * B - 1)]

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def compute_contrastive_loss(a1, a2, a3, temp=0.1):
    """
    计算对比损失：比较a1与a2的距离 vs a1与a3的距离
    
    Args:
        a1: [b, t, c] - 第一个批次
        a2: [b, t, c] - 第二个批次（正样本）
        a3: [b, t, c] - 第三个批次（负样本）
        temp: 温度参数
    
    Returns:
        loss: 对比损失
    """
    
    # 展平矩阵
    a1_flat = a1.reshape(a1.size(0), -1)  # [b, t*c]
    a2_flat = a2.reshape(a2.size(0), -1)  # [b, t*c]
    a3_flat = a3.reshape(a3.size(0), -1)  # [b, t*c]
    
    # 计算a1与a2的相似度（正样本）
    sim_pos = torch.cosine_similarity(a1_flat, a2_flat, dim=1)  # [b]
    
    # 计算a1与a3的相似度（负样本）
    sim_neg = torch.cosine_similarity(a1_flat, a3_flat, dim=1)  # [b]
    
    # 构建对比损失
    # 对于每个样本，我们希望sim_pos尽可能大，sim_neg尽可能小
    # 即：sim_pos - sim_neg 应该尽可能大
    
    # 使用交叉熵损失的形式
    # 将正样本和负样本的相似度组合成logits
    logits = torch.stack([sim_pos, sim_neg], dim=1) / temp  # [b, 2]
    
    # 标签：每个样本的正样本是第一个（sim_pos）
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss

def compute_pairwise_contrastive_loss(a1, a2, a3, temp=0.1):
    """
    计算成对对比损失：a1与a2的相似度 vs a1与a3的相似度
    
    Args:
        a1: [b, t, c] - 第一个批次
        a2: [b, t, c] - 第二个批次（正样本）
        a3: [b, t, c] - 第三个批次（负样本）
    
    Returns:
        loss: 对比损失
    """
    
    # 展平
    a1_flat = a1.reshape(a1.size(0), -1)  # [b, t*c]
    a2_flat = a2.reshape(a2.size(0), -1)  # [b, t*c]
    a3_flat = a3.reshape(a3.size(0), -1)  # [b, t*c]
    
    # 计算相似度
    sim_pos = torch.cosine_similarity(a1_flat, a2_flat, dim=1)  # [b]
    sim_neg = torch.cosine_similarity(a1_flat, a3_flat, dim=1)  # [b]
    
    # 计算损失：希望正样本相似度大于负样本相似度
    # 可以使用margin loss或者对比损失
    margin = 0.1  # 可调节的间隔
    loss = F.relu(sim_neg - sim_pos + margin).mean()
    
    return loss

def compute_triplet_loss(a1, a2, a3, margin=0.1):
    """
    计算三元组损失：||a1-a2||^2 - ||a1-a3||^2 < margin
    
    Args:
        a1: [b, t, c] - 查询样本
        a2: [b, t, c] - 正样本
        a3: [b, t, c] - 负样本
    
    Returns:
        loss: 三元组损失
    """
    a1 = F.normalize(a1, p=2, dim=-1)
    a2 = F.normalize(a2, p=2, dim=-1)
    a3 = F.normalize(a3, p=2, dim=-1)
    # 展平
    a1_flat = a1.reshape(a1.size(0), -1)  # [b, t*c]
    a2_flat = a2.reshape(a2.size(0), -1)  # [b, t*c]
    a3_flat = a3.reshape(a3.size(0), -1)  # [b, t*c]
    
    # 计算欧氏距离的平方
    dist_pos = torch.sum((a1_flat - a2_flat) ** 2, dim=1)  # [b]
    dist_neg = torch.sum((a1_flat - a3_flat) ** 2, dim=1)  # [b]
    
    # 三元组损失
    loss =  torch.clamp(F.relu(dist_pos - dist_neg + margin)+ 1e-8, max=10.0).mean()

    # loss = torch.clamp(loss, max=10.0)
    
    return loss

def compute_advanced_contrastive_loss(a1, a2, a3, temp=0.1):
    """
    高级对比损失：考虑所有样本对
    
    Args:
        a1: [b, t, c] - 第一个批次
        a2: [b, t, c] - 第二个批次（正样本）
        a3: [b, t, c] - 第三个批次（负样本）
    
    Returns:
        loss: 对比损失
    """
    
    # 展平
    a1_flat = a1.reshape(a1.size(0), -1)  # [b, t*c]
    a2_flat = a2.reshape(a2.size(0), -1)  # [b, t*c]
    a3_flat = a3.reshape(a3.size(0), -1)  # [b, t*c]
    
    # 计算所有样本对的相似度矩阵
    # 这里我们计算每对样本的相似度
    sim_pos = torch.cosine_similarity(a1_flat, a2_flat, dim=1)  # [b]
    sim_neg = torch.cosine_similarity(a1_flat, a3_flat, dim=1)  # [b]
    
    # 使用对比损失函数
    # 将正样本和负样本的相似度作为logits
    logits = torch.stack([sim_pos, sim_neg], dim=1) / temp  # [b, 2]
    
    # 标签：每个样本的正样本是第一个（sim_pos）
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # 计算损失
    loss = F.cross_entropy(logits, labels)
    
    return loss

class BatchShuffleContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, temperature=1.0, reduction='mean', max_loss=100.0):
        super(BatchShuffleContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.reduction = reduction
        self.max_loss = max_loss
    
    def forward(self, a1, a2, a3):
        """
        a1: [B, T, C] - 原始序列
        a2: [B, T, C] - 与a1相关的序列
        a3: [B, T, C] - a1在batch维度上随机打乱得到的序列
        """
        # 输入验证
        assert a1.shape == a2.shape == a3.shape, "所有输入张量维度必须相同"
        
        # 确保输入是浮点数
        a1 = a1.float()
        a2 = a2.float()
        a3 = a3.float()
        
        # L2归一化（防止数值溢出）
        a1_norm = F.normalize(a1, p=2, dim=-1)
        a2_norm = F.normalize(a2, p=2, dim=-1)
        a3_norm = F.normalize(a3, p=2, dim=-1)
        
        # 计算相似度
        # a1和a2应该相似（正样本对）
        sim_a1_a2 = torch.sum(a1_norm * a2_norm, dim=-1)  # [B, T]
        # a1和a3应该不相似（负样本对）
        sim_a1_a3 = torch.sum(a1_norm * a3_norm, dim=-1)  # [B, T]
        
        # 平均每个位置的相似度
        sim_a1_a2_mean = sim_a1_a2.mean()
        sim_a1_a3_mean = sim_a1_a3.mean()
        
        # 计算损失：希望sim_a1_a2 > sim_a1_a3
        # 使用margin确保足够的间隔
        diff = sim_a1_a3_mean - sim_a1_a2_mean + self.margin
        
        # 使用稳定的损失函数
        # 方式1：使用softplus函数
        loss = F.softplus(-diff)  # 等价于log(1 + exp(-diff))
        
        # 方式2：如果需要更严格的约束，可以使用
        # loss = F.relu(diff)
        
        # 方式3：最稳健的方式
        # loss = torch.max(torch.tensor(0.0, device=a1.device), diff)
        
        # 防止loss过大
        loss = torch.clamp(loss, max=self.max_loss)
        
        # 添加一些额外的稳定性措施
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=a1.device, requires_grad=True)
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.expand_as(sim_a1_a2)  # 返回原始形状用于其他用途
