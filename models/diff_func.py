import torch
import torch.nn as nn
import models.utils as utils


class ODEFunc(nn.Module):
    def __init__(self, h_dims, h_trans_dims=100, h_trans_layers=1, nonlinear=nn.Tanh, final_nonlinear=None, device=torch.device("cpu")):
        super(ODEFunc, self).__init__()
        self.h_dims = h_dims
        self.h_trans_dims = h_trans_dims
        self.h_trans_layers = h_trans_layers
        self.device = device
        self.using = "ODE_RNN"

        layers = [nn.Linear(h_dims, h_trans_dims)]
        for _ in range(h_trans_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(h_trans_dims, h_trans_dims))
        layers.append(nonlinear())
        layers.append(nn.Linear(h_trans_dims, h_dims))
        self.ode_func = nn.Sequential(*layers).to(device)
        utils.init_network_weights(self.ode_func)
        if final_nonlinear is not None:
            self.final_nonlinear = final_nonlinear()
        else:
            self.final_nonlinear = None

    def extra_repr(self):
        return "hidden_channels: {}, hidden_trans_channels: {}, hidden_trans_layers: {}" \
               "".format(self.h_dims, self.h_trans_dims, self.h_trans_layers)

    def forward(self, t, h, backwards=False):
        func = self.ode_func(h)
        if self.final_nonlinear is not None:
            func = self.final_nonlinear(func)
        if backwards:  # 同一方程，反向求解的时候需要backwards
            func = -func
        return func


class CDEFunc(nn.Module):
    def __init__(self, x_dims, h_dims, h_trans_dims=100, h_trans_layers=1, nonlinear=nn.ReLU, final_nonlinear=nn.Tanh, device=torch.device("cpu")):
        super(CDEFunc, self).__init__()
        self.x_dims = x_dims
        self.h_dims = h_dims
        self.h_trans_dims = h_trans_dims
        self.h_trans_layers = h_trans_layers
        self.device = device
        self.using = "CDE"

        layers = [nn.Linear(h_dims, h_trans_dims)]
        for _ in range(h_trans_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(h_trans_dims, h_trans_dims))
        layers.append(nonlinear())
        layers.append(nn.Linear(h_trans_dims, h_dims * x_dims))
        self.cde_func = nn.Sequential(*layers).to(device)
        utils.init_network_weights(self.cde_func)
        if final_nonlinear is not None:
            self.final_nonlinear = final_nonlinear()
        else:
            self.final_nonlinear = None

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_trans_channels: {}, hidden_trans_layers: {}" \
               "".format(self.x_dims, self.h_dims, self.h_trans_dims, self.h_trans_layers)

    def forward(self, t, h):
        hs = self.cde_func(h)
        hs = hs.view(*h.shape[:-1], self.h_dims, self.x_dims)
        if self.final_nonlinear is not None:
            hs = self.final_nonlinear(hs)  # 最终采用tanh说明输出在(0, 1)中
        return hs