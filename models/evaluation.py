###########################
# Modified from: https://github.com/YuliaRubanova/latent_ode by Kirin Ciao
###########################
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import Independent
import models.utils as utils


def get_log_likelihood(y_pred, y_truth, gaussian_likelihood_std, y_mask=None):
    # y_pred shape: [(sample_hs,) batch_size, time_points, y_dims]
    # y_truth shape: [batch_size, time_points, data_dims]
    if len(y_pred.shape) > 3:
        y_truth = y_truth.repeat(y_pred.shape[0], 1, 1, 1)
    elif len(y_pred.shape) == 3:
        # add additional dimension for sample_hs
        y_pred = y_pred.unsqueeze(0)

    if len(y_truth.shape) == 3:
        # add additional dimension for sample_hs
        y_truth = y_truth.unsqueeze(0)
    elif len(y_truth.shape) == 2:
        # add additional dimension for sample_hs and time_points
        y_truth = y_truth.unsqueeze(0).unsqueeze(2)

    sample_hs, batch_size, time_points, y_dims = y_pred.shape

    mu_2d = y_pred.reshape(sample_hs * batch_size, time_points * y_dims)
    assert torch.Size([sample_hs, batch_size, time_points, y_dims]) == y_truth.shape
    data_2d = y_truth.reshape(sample_hs * batch_size, time_points * y_dims)
    if y_mask is None:
        log_likelihood = compute_log_likelihood(mu_2d, data_2d, gaussian_likelihood_std)
    else:
        if (len(y_mask.shape) == 3):
            y_mask = y_mask.unsqueeze(0)
        assert torch.Size([sample_hs, batch_size, time_points, y_dims]) == y_mask.shape
        mask_2d = y_mask.reshape(sample_hs * batch_size, time_points * y_dims)
        def func(mu_2d, data_2d): return compute_log_likelihood(mu_2d, data_2d, gaussian_likelihood_std)
        log_likelihood = compute_masked_eval(mu_2d, data_2d, mask_2d, func)

    log_likelihood = log_likelihood.reshape(sample_hs, batch_size)
    log_likelihood = torch.mean(log_likelihood, 1)  # batch内各sample内取平均
    # shape: [sample_hs] or [1]
    return log_likelihood


def get_mse(y_pred, y_truth, y_mask=None):
    # y_pred shape: [(sample_hs,) batch_size, time_points, y_dims]
    # y_truth shape: [batch_size, time_points, data_dims]
    if len(y_pred.shape) > 3:
        y_truth = y_truth.repeat(y_pred.shape[0], 1, 1, 1)
    elif len(y_pred.shape) == 3:
        # add additional dimension for sample_hs
        y_pred = y_pred.unsqueeze(0)

    if len(y_truth.shape) == 3:
        # add additional dimension for sample_hs
        y_truth = y_truth.unsqueeze(0)
    elif len(y_truth.shape) == 2:
        # add additional dimension for sample_hs and time_points
        y_truth = y_truth.unsqueeze(0).unsqueeze(2)

    sample_hs, batch_size, time_points, y_dims = y_pred.shape

    mu_2d = y_pred.reshape(sample_hs * batch_size, time_points * y_dims)
    assert torch.Size([sample_hs, batch_size, time_points, y_dims]) == y_truth.shape
    data_2d = y_truth.reshape(sample_hs * batch_size, time_points * y_dims)
    # Shape after permutation: [sample_hs, batch_size, time_points, y_dims]
    if y_mask is None:
        mse = compute_mse(mu_2d, data_2d)
    else:
        if (len(y_mask.shape) == 3):
            y_mask = y_mask.unsqueeze(0)
        assert torch.Size([sample_hs, batch_size, time_points, y_dims]) == y_mask.shape
        mask_2d = y_mask.reshape(sample_hs * batch_size, time_points * y_dims)
        mse = compute_masked_eval(mu_2d, data_2d, mask_2d, compute_mse)

    # shape: []
    return mse


def compute_masked_eval(mu_2d, data_2d, mask_2d, eval_func):
    res = []
    for i in range(data_2d.shape[0]):
        data_1d_masked = torch.masked_select(data_2d[i, :], mask_2d[i, :])
        mu_1d_masked = torch.masked_select(mu_2d[i, :], mask_2d[i, :])
        eval_for_a_sample = eval_func(mu_1d_masked, data_1d_masked)
        res.append(eval_for_a_sample)
    res = torch.stack(res, 0)
    return res


def compute_log_likelihood(mu, data, gaussian_likelihood_std):
    # gaussian_likelihood_std 全部用在这
    n_data_points = mu.shape[-1]
    # Normal以预测值为均值的正态分布，Independent使batch中各条曲线相互独立，从而算出sample_h * batch_size个维度的likelihood
    if n_data_points > 0:
        gaussian = Independent(Normal(loc=mu, scale=gaussian_likelihood_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros((1), device=utils.get_device(data)).squeeze()
    return log_prob


def compute_mse(mu, data):
    data_points = mu.shape[-1]

    if data_points > 0:
        mse_loss = nn.MSELoss()(mu, data)
    else:
        mse_loss = torch.zeros((1), device=utils.get_device(data)).squeeze()
    return mse_loss