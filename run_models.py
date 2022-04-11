import os
import gc
from random import SystemRandom
import torch
from torch.distributions.normal import Normal
from tqdm import tqdm
import models.utils as utils
from models.recurrent import Recurrent
from models.seq2seq import Seq2Seq
from models.vae import VAE
from models.diff_func import ODEFunc, CDEFunc
from models.diffeq_solver import DiffeqSolver
import matplotlib
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
matplotlib.use("Agg")


def define_args():
    # 默认设置
    Args = {
        "no_save": False,
        # Data
        "dataset": "GoogleStock",
        "x_data_with_time": False,
        "load_data": True,
        "x_y_points": None,
        "x_points": None,
        "y_dim_list": None,
        "y_delay": 0,
        "train_fraq": 0.8,
        "val_fraq": None,
        "shuffle": False,
        # Model
        "arch": "Seq2Seq",
        "using": "ODE_RNN",
        "load_ckpt": None,
        # ODE Solver
        "ode_time": "No",
        "method": "dopri5",
        "rtol": 1e-3,
        "atol": 1e-4,
        # Latent_ODE
        "h_trans_layers": 1,
        "n_gru_units": 100,
        "n_out_units": 100,
        "seed": 2021,
        # Train Hyperparam
        "max_epochs": 300,
        "patience_for_no_better_epochs": 30,
        "lr": 1e-2,
        "kl_coef": 1.,
        "batch_size": 50,  # 多少段时间序列组成一组
        "gaussian_likelihood_std": 0.01,
        "missing_rate": None,
        "progress_train": False,
        "device": torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0"),
        # Test and Plot
        "save_res_for_epochs": 1,
        "test_for_epochs": 1,
        "save_log": True,
        "save_ckpt": True,
        "save_fig_per_test": 1,
        "y_data_color": "gray",
        "x_data_color": "dodgerblue",
        "y_pred_color": "orange"
    }

    argschange_note = utils.argschange(Args)  # 在这里记录命令行参数对Args的变动

    if Args["no_save"]:
        Args["save_res_for_epochs"] = None
        Args["save_log"] = False
        Args["save_ckpt"] = False
        Args["save_fig_per_test"] = 0

    if Args["using"] == "ODE_RNN":
        Args["produce_intercoeffs"] = False
    else:
        Args["produce_intercoeffs"] = True

    if Args["dataset"] == "GoogleStock":
        Args["csv_list"] = ["data"]
        Args["continue_params"] = ["No", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        Args["batch_size"] = 200
        Args["lr"] = 0.001 * (Args["batch_size"] / 32)
        Args["x_y_points"] = 25
        Args["x_points"] = 24
        Args["stride"] = 1
        Args["train_fraq"] = 0.7
        Args["val_fraq"] = 0.15
        Args["shuffle"] = True
        Args["missing_rate"] = 0.7
        Args["ode_time"] = "Mean"
        Args["h_trans_layers"] = 2
        Args["save_fig_per_test"] = 0  # 不画图

    return Args, argschange_note


def get_model(Args, x_dims, y_dims):
    # 求解器从ruler换成dopri5
    if Args["arch"] == "Recurrent":
        diffeq_solver = get_diffeq_solver(Args, x_dims)
        model = Recurrent(
            x_dims=x_dims,
            y_dims=y_dims,
            n_gru_units=Args["n_gru_units"],
            n_out_units=Args["n_out_units"],
            diffeq_solver=diffeq_solver,
            gaussian_likelihood_std=Args["gaussian_likelihood_std"]
        )
    elif Args["arch"] == "Seq2Seq":
        enc_diffeq_solver = get_diffeq_solver(Args, x_dims)
        dec_diffeq_solver = get_diffeq_solver(Args, x_dims)
        model = Seq2Seq(
            x_dims=x_dims,
            y_dims=y_dims,
            n_gru_units=Args["n_gru_units"],
            n_out_units=Args["n_out_units"],
            enc_diffeq_solver=enc_diffeq_solver,
            dec_diffeq_solver=dec_diffeq_solver,
            gaussian_likelihood_std=Args["gaussian_likelihood_std"]
        )
    elif Args["arch"] == "VAE":
        enc_diffeq_solver = get_diffeq_solver(Args, x_dims, h_dims=20, method="euler")
        dec_diffeq_solver = get_diffeq_solver(Args, x_dims)
        model = VAE(
            x_dims=x_dims,
            y_dims=y_dims,
            h_prior=Normal(torch.Tensor([0.0]).to(
                Args["device"]), torch.Tensor([1.]).to(Args["device"])),  # h的先验分布
            n_gru_units=Args["n_gru_units"],
            n_out_units=Args["n_out_units"],
            enc_diffeq_solver=enc_diffeq_solver,
            dec_diffeq_solver=dec_diffeq_solver,
            gaussian_likelihood_std=Args["gaussian_likelihood_std"]
        )
    else:
        raise NotImplementedError

    return model


def get_diffeq_solver(Args, x_dims, h_dims=None, h_trans_layers=None, method=None, rtol=None, atol=None):
    h_dims = h_dims if h_dims is not None else x_dims * 2
    h_trans_layers = h_trans_layers if h_trans_layers is not None else Args["h_trans_layers"]
    method = method if method is not None else Args["method"]
    rtol = rtol if rtol is not None else Args["rtol"]
    atol = atol if atol is not None else Args["atol"]
    if Args["using"] == "ODE_RNN":
        return DiffeqSolver(ODEFunc(h_dims, h_trans_layers=h_trans_layers, device=Args["device"]),
                            method, rtol, atol)
    elif Args["using"] == "CDE":
        return DiffeqSolver(CDEFunc(x_dims, h_dims, h_trans_dims=h_dims, h_trans_layers=h_trans_layers, device=Args["device"]),
                            method, rtol, atol)
    else:
        raise NotImplementedError


def prepare_to_train(Args):
    # 数据集
    train_dict, val_dict, test_dict, x_dims, y_dims = utils.get_data(Args)
    # 状态读取
    if Args["load_ckpt"] is not None and os.path.isfile(Args["load_ckpt"]):
        model, optimizer, scheduler, pre_points, experimentID = utils.load_checkpoint(Args["load_ckpt"])
        model.to(Args["device"])
    else:
        # 模型、优化器、LR的值改动
        model = get_model(Args, x_dims, y_dims)
        optimizer = torch.optim.Adamax(model.parameters(), lr=Args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        if Args["progress_train"]:  # TODO: curriculum learning for encoder-decoder
            pre_points = 10
        else:
            pre_points = None
        experimentID = str(int(SystemRandom().random()*1000000))
    return model, optimizer, scheduler, pre_points, experimentID, train_dict, val_dict, test_dict


def test_and_plot(Args, model, test_dataloader, kl_coef, experimentID, res_files, fig_saveat, logger, epoch):
    model.eval()
    with torch.no_grad():
        # 保存最佳结果
        test_res, res_dict = utils.store_best_results(Args, model, test_dataloader, kl_coef, experimentID, res_files)
        # 可视化
        if not Args["produce_intercoeffs"] and fig_saveat is not None:
            def to_np(x): return x.detach().cpu().numpy() if x.is_cuda else x.detach().numpy()  # 设备
            batch_list = next(iter(test_dataloader))  # 最大画图数量不超过batch_size
            batch_dict = utils.time_sync_batch_dict(batch_list, Args["ode_time"])
            if Args["arch"] == "Recurrent" or Args["arch"] == "Seq2Seq":
                y_pred = model(batch_dict["y_time"], batch_dict["x_data"], batch_dict["x_time"], batch_dict["x_mask"])
            else:
                _, info = model(batch_dict["y_time"], batch_dict["x_data"], batch_dict["x_time"], batch_dict["x_mask"])
                y_pred = info["y_pred_mean"]
            # y_pred shape: [batch_size, time_points, data_dims]
            # TODO: 一维，自回归情况，且不能用于插值
            for k in range(Args["save_fig_per_test"]):
                plt.clf()
                plt.plot(to_np(batch_dict["y_time"]), to_np(batch_dict["y_data"][k, :, 0]), color=Args["y_data_color"], linestyle="--")
                plt.scatter(to_np(batch_dict["x_time"]), to_np(batch_dict["x_data"][k, :, 0]), color=Args["x_data_color"], marker="s")
                plt.plot(to_np(batch_dict["y_time"]), to_np(y_pred[k, :, 0]), color=Args["y_pred_color"])
                plt.savefig(fig_saveat + "/" + str(epoch) + "_" + str(k) + ".jpg")

        # 保存输出字段
        output_str = "Test at Epoch: %4d | Loss: %f | MSE: %f" % (epoch, test_res["loss"].item(), test_res["mse"].item())
        logger.info(output_str)
        logger.info(res_dict)

    return output_str


if __name__ == "__main__":
    Args, argschange_note = define_args()
    model, optimizer, scheduler, pre_points, experimentID, train_dict, val_dict, test_dict = prepare_to_train(Args)
    logger, res_files, train_res_csv, val_res_csv, fig_saveat, ckpt_saveat = utils.get_logger_and_save(model, Args, argschange_note, experimentID)

    # 开始训练
    # 采样部分的点，progress_training逐渐预测多点
    train_dataloader = utils.masked_dataloader(train_dict, Args, Args["missing_rate"], pre_points, "train")
    val_dataloader = utils.masked_dataloader(val_dict, Args, Args["missing_rate"], None, "val")
    test_dataloader = utils.masked_dataloader(test_dict, Args, Args["missing_rate"], None, "test")

    epoch_test = max(1, Args["test_for_epochs"])
    pbar = tqdm(total=epoch_test)

    # 用于保存最佳模型和决定是否退出训练
    best_metric = torch.tensor([torch.inf], device=Args["device"])
    best_metric_epoch = 0
    stop_training = False

    for epoch in range(1, Args["max_epochs"] + 1):
        if stop_training:
            break
        model.train()  # 预备使用BatchNorm1d(对时间序列，暂时不需要)
        kl_coef = utils.update_kl_coef(Args["kl_coef"], epoch)
        pbar.set_description("Epoch [%4d / %4d]" % (epoch, Args["max_epochs"]))

        # 对每个batch
        for train_batch_list in train_dataloader:
            optimizer.zero_grad()
            batch_dict = utils.time_sync_batch_dict(train_batch_list, Args["ode_time"])
            train_res = model.compute_loss_one_batch(batch_dict, kl_coef)
            gc.collect()
            # 反向传播
            train_res["loss"].backward()
            optimizer.step()
            gc.collect()
            pbar.set_postfix(Loss=train_res["loss"].item(), MSE=train_res["mse"].item())

        # 到达一个epoch
        pbar.update(1)
        if train_res_csv is not None and epoch % Args["save_res_for_epochs"] == 0:
            train_res_csv.write(train_res, epoch)
        # 更新超参数
        if val_dataloader is None:  # 使用训练集更新
            scheduler.step(train_res["loss"].item())
            if train_res["mse"] * 1.0001 < best_metric:
                best_metric = train_res["mse"]
                best_metric_epoch = epoch
                # 检查点
                checkpoint = (model, optimizer, scheduler, pre_points, experimentID)
                utils.save_checkpoint(ckpt_saveat, checkpoint, name="best_for_train")
        else:  # 使用验证集更新
            val_res = utils.compute_loss_all_batches(model, val_dataloader, kl_coef, ode_time=Args["ode_time"])
            scheduler.step(val_res["loss"].item())
            if val_res_csv is not None and epoch % Args["save_res_for_epochs"] == 0:
                val_res_csv.write(val_res, epoch)
            if val_res["mse"] * 1.0001 < best_metric:
                best_metric = val_res["mse"]
                best_metric_epoch = epoch
                # 检查点
                checkpoint = (model, optimizer, scheduler, pre_points, experimentID)
                utils.save_checkpoint(ckpt_saveat, checkpoint, name="best_for_val")

        # curriculum learning
        if pre_points is not None:
            pre_points += 10
            train_dataloader = utils.masked_dataloader(train_dict, Args, Args["missing_rate"], pre_points, "train")
        # 超出对应epoch，停止训练
        if Args["patience_for_no_better_epochs"] is not None:
            if epoch > best_metric_epoch + Args["patience_for_no_better_epochs"]:
                pbar.close()
                tqdm.write("No better metrics than %f for %d epochs. Stop training." % (best_metric.item(), Args["patience_for_no_better_epochs"]))
                output_str = test_and_plot(Args, model, test_dataloader, kl_coef, experimentID, res_files, fig_saveat, logger, epoch)  # 此时进行一次测试
                tqdm.write(output_str)
                stop_training = True

        # 测试集
        if epoch % epoch_test == 0 or epoch == Args["max_epochs"]:
            output_str = test_and_plot(Args, model, test_dataloader, kl_coef, experimentID, res_files, fig_saveat, logger, epoch)
            pbar.close()
            if epoch != Args["max_epochs"]:  # 开新的进度条
                pbar = tqdm(total=epoch_test)
            tqdm.write(output_str)  # 展示上一次测试结果
