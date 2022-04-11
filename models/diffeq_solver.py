import torch.nn as nn
from torchdiffeq import odeint
from torchcde import cdeint


class DiffeqSolver(nn.Module):
    def __init__(self, diff_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolver, self).__init__()

        self.diff_func = diff_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.ode_method = method

    def forward(self, h0, t, X=None):
        if self.diff_func.using == "ODE_RNN":
            if len(h0.shape) == 2:
                batch_size, h_dims = h0.shape[0], h0.shape[1]
                hs_pred = odeint(self.diff_func, h0, t,
                                 rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
                hs_pred = hs_pred.permute(1, 0, 2)  # 将time_points对应维度移到倒数第二位
            else:
                sample_hs, batch_size, h_dims = h0.shape[0], h0.shape[1], h0.shape[2]
                hs_pred = odeint(self.diff_func, h0, t, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
                hs_pred = hs_pred.permute(1, 2, 0, 3)  # 将time_points对应维度移到倒数第二位
                assert hs_pred.shape[0] == sample_hs
            assert hs_pred.shape[-3] == batch_size
            assert hs_pred.shape[-1] == h_dims

        elif self.diff_func.using == "CDE":
            kwargs = {}
            # kwargs["rtol"] = self.odeint_rtol
            # kwargs['atol'] = self.odeint_atol
            # kwargs["method"] = self.ode_method
            kwargs["method"] = "rk4"
            kwargs["options"] = {}
            kwargs["options"]["step_size"] = (t[1:] - t[:-1]).min().item()
            hs_pred = cdeint(X, self.diff_func, h0, t, adjoint=True, backend="torchdiffeq", **kwargs)
            # 不需要转置，输出[batch_size, time_points, h_dims]或[sample_hs, batch_size, time_points, h_dims]

        else:
            raise NotImplementedError

        return hs_pred  # 输出[batch_size, time_points, h_dims]或[sample_hs, batch_size, time_points, h_dims]
