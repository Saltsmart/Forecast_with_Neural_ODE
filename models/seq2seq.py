import torch
import torch.nn as nn
from models.encoder_decoder import ODE_GRU_Encoder, ODE_Decoder
from models.evaluation import get_log_likelihood, get_mse


class Seq2Seq(nn.Module):
    def __init__(self, x_dims, y_dims,
                 n_gru_units=100, n_out_units=100,
                 enc_diffeq_solver=None, dec_diffeq_solver=None,
                 gaussian_likelihood_std=0.02):

        super(Seq2Seq, self).__init__()
        enc_h_dims = enc_diffeq_solver.diff_func.h_dims
        dec_h_dims = dec_diffeq_solver.diff_func.h_dims
        assert enc_diffeq_solver.diff_func.device == dec_diffeq_solver.diff_func.device
        self.device = enc_diffeq_solver.diff_func.device
        self.gaussian_likelihood_std = torch.tensor([gaussian_likelihood_std], device=self.device)

        self.encoder = ODE_GRU_Encoder(
            x_dims=x_dims,
            h_dims=enc_h_dims,
            n_gru_units=n_gru_units,
            h_last_dim=dec_h_dims,
            enc_diffeq_solver=enc_diffeq_solver,
            device=self.device)

        self.decoder = ODE_Decoder(y_dims, dec_h_dims, n_out_units, dec_diffeq_solver, device=self.device)

    def compute_loss_one_batch(self, batch_dict, kl_coef=1.):
        # 预测整个y序列，包括观测到的与未观测到的
        y_pred = self.forward(batch_dict["y_time"], batch_dict["x_data"], batch_dict["x_time"], batch_dict["x_mask"])

        # 计算指标，batch_dict["y_mask"]如果都是True，那就应该输入None
        log_likelihood = get_log_likelihood(y_pred, batch_dict["y_data"], self.gaussian_likelihood_std, None if batch_dict["y_mask"].all() else batch_dict["y_mask"]).squeeze()
        mse = get_mse(y_pred, batch_dict["y_data"], None if batch_dict["y_mask"].all() else batch_dict["y_mask"])
        loss = - log_likelihood.squeeze()  # -log_likelihood反向传播

        # batch内取平均
        results = {}
        results["loss"] = loss
        results["likelihood"] = log_likelihood.detach()
        results["mse"] = mse.detach()
        results["kl_coef"] = kl_coef
        results["C_kl"] = 0.
        results["C_std"] = 0.

        return results

    def forward(self, y_time, x_data, x_time, x_mask=None):
        # 完成序列预测
        if x_mask is not None:
            x = x_data * x_mask
        else:
            x = x_data

        if len(y_time.shape) < 1:
            y_time = y_time.unsqueeze(0)

        # encoder
        hs = self.encoder.run_to_last_point(x, x_time, return_latents=True)

        decoder_begin_hi = hs[:, -1, :]  # 将decoder初始点设为x_time最后一点
        y_time = torch.cat((x_time[-1:], y_time))

        # decoder
        y_pred = self.decoder(decoder_begin_hi, y_time)[:, 1:, :]  # 继续向下解ode，利用decoder解出预测值

        return y_pred
