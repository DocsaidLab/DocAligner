**[English](./README.md)** | [中文](./README_cn.md)

# Training Overview

We have a method of configuring parameter files that allows for customization of the training process. This file is located at `config/***.yaml`. You can use this file to adjust various settings to suit your specific model architecture, dataset, and training requirements.

## Environment Requirements

Please ensure that your environment meets the following requirements:

- Python 3.8 or higher version

## Executing Training

Run the following command to start the training process:

```bash
python trainer.py --cfg_name lcnet_100_fuse_all_edge
```

This command will initiate model training using the `trainer.py` script along with the specified configuration name `lcnet_100_fuse_all_edge`. Ensure that your current working directory is the one containing the script when running this command.

## Configuration Customization

Customizing the training process involves adjusting various settings to match your specific model architecture, dataset, and training requirements. The following sections provide detailed information on how to customize various aspects of training using the configurations available in `config/***.yaml`:

### Trainer Settings

Adjust trainer settings to control the training process using PyTorch Lightning:

- `max_epochs`: Set the maximum number of training epochs.
- `precision`: Define the precision of computations (e.g., 32-bit).
- `val_check_interval`: Set the interval for validation checks during training.
- `gradient_clip_val`: Set the threshold for gradient clipping.
- `accumulate_grad_batches`: Define the number of batches for accumulating gradients.
- `accelerator`: Choose the type of accelerator for training.
- `devices`: Specify the number of devices (GPUs) to use.

For more information, refer to [PyTorch Lightning Trainer Documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

### Model Settings

Configure model settings according to your custom architecture:

- `name`: Specify the name of the model.
- `backbone`: Customize the backbone architecture and its parameters.
- `neck`: Adjust settings for the neck architecture.
- `head`: Define the head architecture parameters suitable for your model.

### Dataset

Adjust dataset settings based on your custom dataset:

- `train_options`: Set options for the training dataset, including path, augmentation, and size.
- `valid_options`: Similarly to the training dataset, configure options for the validation dataset.

### DataLoader

Fine-tune data loader settings using PyTorch's DataLoader:

- `train_options`: Set options for the training data loader, such as the number of worker processes, random shuffling, and batch processing.
- `valid_options`: Configure similar settings for the validation data loader.

### Optimizer

Customize optimizer settings using PyTorch's optimizers:

- `name`: Specify the name of the optimizer (e.g., AdamW).
- `options`: Define options for the optimizer, such as learning rate, weight decay, and betas.

### Learning Rate Scheduler

Adjust learning rate scheduler settings using PyTorch's schedulers:

- `name`: Specify the name of the scheduler (e.g., MultiStepLRWarmUp).
- `options`: Configure specific options for the scheduler.

### Callbacks

Enhance training using callbacks with PyTorch Lightning:

- Configure various callbacks, such as `ModelCheckpoint` and `LearningRateMonitor`, to improve training efficiency and tracking.

### Logger

Monitor training progress using PyTorch Lightning's loggers:

- Configure `TensorBoardLogger` to visualize training metrics with TensorBoard.
