import torch
import torch.nn as nn
import models.utils as utils
from models.recurrent import GRU_with_std


class ODE_GRU_Encoder(nn.Module):
    def __init__(self, x_dims, h_dims,
                 n_gru_units=100, h_last_dim=None, enc_diffeq_solver=None, device=torch.device("cpu"), GRU_update=None):

        super(ODE_GRU_Encoder, self).__init__()

        if h_last_dim is None:
            self.h_last_dim = h_dims
        else:
            self.h_last_dim = h_last_dim

        if GRU_update is None:
            self.GRU_update = GRU_with_std(h_dims, x_dims, n_units=n_gru_units, device=device)
        else:
            self.GRU_update = GRU_update.to(device)

        self.enc_diffeq_solver = enc_diffeq_solver
        self.h_dims = h_dims
        self.x_dims = x_dims
        self.device = device

        self.transform_h_last = utils.create_net(self.h_dims * 2, self.h_last_dim * 2, device=device)
        utils.init_network_weights(self.transform_h_last)

    def run_to_prior(self, x_data, x_time):
        assert not torch.isnan(x_time).any()

        batch_size = x_data.shape[0]
        if self.device != utils.get_device(x_data):
            x_data.to(self.device)
        if self.device != utils.get_device(x_time):
            x_time.to(self.device)

        if x_time.numel() == 1:
            prev_hi = torch.zeros((batch_size, self.h_dims)).to(self.device)
            prev_hi_std = torch.zeros((batch_size, self.h_dims)).to(self.device)
            xi = x_data[:, 0, :]
            assert not torch.isnan(xi).any()
            hi_last_mu, hi_last_sigma = self.GRU_update(prev_hi, prev_hi_std, xi)
        else:
            hi_last_mu, hi_last_sigma = self.run_to_last_point(x_data, x_time, return_latents=False)

        hi_last_mu = hi_last_mu.reshape(1, batch_size, self.h_dims)
        hi_last_sigma = hi_last_sigma.reshape(1, batch_size, self.h_dims)

        # VAE2：通过encoder(x,hi_last_mu,hi_last_sigma)产生对应的mu和sigma
        hi_last_mu, hi_last_sigma = utils.split_last_dim(self.transform_h_last(torch.cat((hi_last_mu, hi_last_sigma), -1)))
        hi_last_sigma = hi_last_sigma.abs()

        return hi_last_mu, hi_last_sigma

    def run_to_last_point(self, x_data, x_time, return_latents=False):
        # VAE1：同时生成最后一个观测点的平均值hi和标准差hi_std
        batch_size = x_data.shape[0]
        if self.device != utils.get_device(x_data):
            x_data.to(self.device)
        if self.device != utils.get_device(x_time):
            x_time.to(self.device)

        prev_hi = torch.zeros((batch_size, self.h_dims), device=self.device)
        prev_hi_std = torch.zeros((batch_size, self.h_dims), device=self.device)

        assert not torch.isnan(x_time).any()

        prev_ti, ti = x_time[-1] + 0.01, x_time[-1]
        # 内部网格大小
        interval_length = x_time[-1] - x_time[0]
        minimum_step = interval_length / 50

        hs = []
        time_points_iter = range(0, x_time.numel())

        for i in time_points_iter:
            if (prev_ti - ti) < minimum_step:  # 如果达不到最小步长，则直接线性增加
                time_points = torch.stack((prev_ti, ti))
                inc = self.enc_diffeq_solver.diff_func(prev_ti, prev_hi) * (ti - prev_ti)  # 线性增量

                if torch.isnan(inc).any():
                    print("inc is None!!")
                    print(i)
                    print(self.enc_diffeq_solver.diff_func(prev_ti, prev_hi))
                    print(torch.isnan(prev_hi).any())

                assert not torch.isnan(ti - prev_ti).any()
                assert not torch.isnan(prev_ti).any()
                assert not torch.isnan(prev_hi).any()

                hi_ode = prev_hi + inc

                assert not torch.isnan(hi_ode).any()

            else:
                n_intermediate_tp = max(
                    2, ((prev_ti - ti) / minimum_step).int())
                time_points = utils.linspace_vector(prev_ti, ti, n_intermediate_tp, device=self.device)

                assert not torch.isnan(time_points).any()
                assert not torch.isnan(prev_hi).any()

                hi_ode = self.enc_diffeq_solver(prev_hi, time_points)[:, -1, :]  # 计算ODE，解出对应的latent值来（最后1个）即可

                assert not torch.isnan(hi_ode).any()

            xi = x_data[:, i, :]

            assert not torch.isnan(hi_ode).any()
            assert not torch.isnan(prev_hi_std).any()

            if torch.isnan(xi).any():
                hi, hi_std = hi_ode, prev_hi_std
            else:
                hi, hi_std = self.GRU_update(hi_ode, prev_hi_std, xi)  # 计算GRU

            assert not torch.isnan(hi).any()
            assert not torch.isnan(hi_std).any()

            # prev_hi-(odesolver)->hi_ode-(GRU)->hi->prev_hi
            prev_hi, prev_hi_std = hi, hi_std
            prev_ti, ti = x_time[i], x_time[i - 1]

            if return_latents:
                hs.append(hi)

        if return_latents:
            hs = torch.stack(hs, 1)
            assert not torch.isnan(hs).any()
            return hs
        else:
            assert not torch.isnan(hi).any()
            assert not torch.isnan(hi_std).any()
            return hi, hi_std


class ODE_Decoder(nn.Module):
    # ODE作为decoder
    def __init__(self, y_dims, h_dims, n_out_units, dec_diffeq_solver, device=torch.device("cpu")):
        super(ODE_Decoder, self).__init__()
        self.output_net = utils.create_net(h_dims, y_dims, n_out_units, device=device)
        utils.init_network_weights(self.output_net)

        self.dec_diffeq_solver = dec_diffeq_solver

    def forward(self, h0, t_span):
        ode_sol = self.dec_diffeq_solver(h0, t_span)  # 继续向下解ode
        outputs = self.output_net(ode_sol)
        return outputs
