import gc
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
import models.utils as utils
from models.encoder_decoder import ODE_GRU_Encoder, ODE_Decoder
from models.evaluation import get_log_likelihood, get_mse


class VAE(nn.Module):
    def __init__(self, x_dims, y_dims, h_prior,
                 n_gru_units=100, n_out_units=100,
                 enc_diffeq_solver=None, dec_diffeq_solver=None,
                 gaussian_likelihood_std=0.02):

        super(VAE, self).__init__()
        enc_h_dims = enc_diffeq_solver.diff_func.h_dims
        dec_h_dims = dec_diffeq_solver.diff_func.h_dims
        assert enc_diffeq_solver.diff_func.device == dec_diffeq_solver.diff_func.device
        self.device = enc_diffeq_solver.diff_func.device
        self.gaussian_likelihood_std = torch.tensor([gaussian_likelihood_std], device=self.device)
        self.h_prior = h_prior


        self.encoder = ODE_GRU_Encoder(
            x_dims=x_dims,  # mask直接作用于data
            h_dims=enc_h_dims,
            n_gru_units=n_gru_units,
            h_last_dim=dec_h_dims,
            enc_diffeq_solver=enc_diffeq_solver,
            device=self.device)

        self.decoder = ODE_Decoder(y_dims, dec_h_dims, n_out_units, dec_diffeq_solver, device=self.device)

    def compute_loss_one_batch(self, batch_dict, kl_coef=1., sample_hs=3):
        # 预测整个y序列，包括观测到的与未观测到的
        y_pred, info = self.forward(batch_dict["y_time"], batch_dict["x_data"], batch_dict["x_time"], batch_dict["x_mask"], sample_hs=sample_hs)

        C_mu, C_std, _ = info["C"]
        C_std = C_std.abs()
        C_distr = Normal(C_mu, C_std)

        assert torch.sum(C_std < 0) == 0.

        C_kl = kl_divergence(C_distr, self.h_prior)

        if torch.isnan(C_kl).any():
            print(C_mu)
            print(C_std)
            raise Exception("C_kldiv is Nan!")

        # C_kldiv shape: [sample_hs, batch_size, h_dims] if prior is a mixture of gaussians (KL is estimated)
        # C_kldiv shape: [1, batch_size, h_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [sample_hs]
        # 与IWAE有关：https://zhuanlan.zhihu.com/p/74556487，进行sample_h次采样以获得更好的重构能力

        C_kl = torch.mean(C_kl, (1, 2)).squeeze()

        # 计算指标
        if not batch_dict["y_mask"].all():  # 如果不全是True，说明起作用
            y_mask = batch_dict["y_mask"].repeat(sample_hs, 1, 1, 1)  # 复制mask
        else:
            y_mask = None
        log_likelihood = get_log_likelihood(y_pred, batch_dict["y_data"], self.gaussian_likelihood_std, y_mask)
        mse = get_mse(y_pred, batch_dict["y_data"], y_mask)
        # IWAE loss
        loss = - torch.logsumexp(log_likelihood - kl_coef * C_kl, 0)
        if torch.isnan(loss):
            loss = - torch.mean(log_likelihood - kl_coef * C_kl, 0)

        # batch内取平均
        results = {}
        results["loss"] = loss
        results["likelihood"] = torch.mean(log_likelihood).detach()
        results["mse"] = mse.detach()
        results["kl_coef"] = kl_coef
        results["C_kl"] = C_kl.detach()
        results["C_std"] = torch.mean(C_std).detach()

        return results

    def forward(self, y_time, x_data, x_time, x_mask=None, sample_hs=1):
        # 完成序列预测
        if x_mask is not None:
            x = x_data * x_mask
        else:
            x = x_data
        if len(y_time.shape) < 1:
            y_time = y_time.unsqueeze(0)

        # encoder
        enc_h_last_mu, enc_h_last_sigma = self.encoder.run_to_prior(x, x_time)

        # VAE3：通过正态分布生成decoder的h（这里进行了sample_hs次采样）
        enc_h_last_mean = enc_h_last_mu.repeat(sample_hs, 1, 1)
        enc_h_last_std = enc_h_last_sigma.repeat(sample_hs, 1, 1)
        dec_h_first = utils.sample_standard_gaussian(enc_h_last_mean, enc_h_last_std)
        enc_h_last_sigma = enc_h_last_sigma.abs()
        assert torch.sum(enc_h_last_sigma < 0) == 0.
        gc.collect()
        assert not torch.isnan(y_time).any()
        assert not torch.isnan(dec_h_first).any()
        # 将decoder初始点设为x_time最后一点
        y_time = torch.cat((x_time[-1:], y_time))

        # decoder
        y_pred = self.decoder(dec_h_first, y_time)[:, :, 1:, :]  # 利用decoder解出预测值，用于训练和测试
        y_pred_mean = self.decoder(enc_h_last_mean, y_time)[:, :, 1:, :]  # 利用采样前的值再解出1组取平均，用于画图
        assert not torch.isnan(y_pred).any()
        assert not torch.isnan(y_pred_mean).any()

        info = {
            "C": (enc_h_last_mu, enc_h_last_sigma, dec_h_first),
            "y_pred_mean": torch.mean(y_pred_mean, 0)
        }

        return y_pred, info
