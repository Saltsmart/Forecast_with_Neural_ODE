# How to use Neural ODE in Time Forecasting?

This is a repo for Neural ODE and CDE forecasters.

## Installation and Data Preparation

You should install these necessary packages:

* matplotlib>=3.5.1
* numpy>=1.22.2
* pandas>=1.4.1
* torch>=1.10.2
* torchcde>=0.2.5
* torchdiffeq>=0.2.2
* tqdm>=4.63.0

Then place your own data files in `data/`. Please change `models/utils.py/get_data` function to get **dicts** for train, val (if optional else **None**) and test. These dicts must be formulated by Pytorch tensors as:

```python
"x_time": torch.Size([all_samples, x_time_points, 1])  # Timestamps
"x_data": torch.Size([all_samples, x_time_points, x_dims])  # input data
"x_mask": torch.Size([all_samples, x_time_points, x_dims])  # input mask. Bool tensor.
"y_time": torch.Size([all_samples, y_time_points, 1])  # Timestamps
"y_data": torch.Size([all_samples, y_time_points, x_dims])  # output data
"y_mask": torch.Size([all_samples, y_time_points, x_dims])  # output mask. Bool tensor.
```

## Run commands to train & test

I've implemented a function `models/utils.py/argschange` to change the `Args` dict in `run_models.py` (A simple alternative to `argparse` package!), so you can run commands like:

```
python run_models.py --arch Recurrent --using CDE --no_save
```

Or simply click and run `run_models.py`. This way makes sense because the arguments mentioned above have default values in `Args`.

To use different models, you can set:

```
--arch: "Recurrent", "Seq2Seq" and "VAE" are available.
--using: "ODE_RNN" or "CDE". "CDE" only for "Recurrent" now.
```

`--no_save` argument prevents the model to create experiment results files. Without this, visualization results can be found in `fig/`, result logs in `log/` and checkpoint files in `ckpt/`.

`delete.py` is used to clear files above. You can type:

```
python delete.py --ID 999999
```

In order to clear files for experiment ID 999999, for instance.
