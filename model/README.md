# Model Training

This section of the documentation provides guidance on how to perform model training. Follow the steps below to ensure a smooth training process.

## Requirements

Make sure your environment meets the following requirements:

- Python 3.8 or higher

## Install Dependencies

Before starting the training process, ensure that you have all the required dependencies installed. You can install the necessary Python packages using the following command:

```bash
pip install -r requirements.txt
```

This will install the packages listed in the `requirements.txt` file.

## Running the Training

Execute the following command to start the training process:

```bash
python trainer.py --cfg_name lcnet_100_fuse_all_edge
```

This command will use the `trainer.py` script along with the specified configuration name `lcnet_100_fuse_all_edge` to initiate the model training. Make sure that your current working directory is the one containing the script when running the command.

## Configuration Customization

Customizing the training process involves adjusting various settings to match your specific model architecture, dataset, and training requirements. The following sections provide details on how to customize each aspect of the training using the provided configuration in `config/***.yaml`:

### Trainer Settings

Adjust the trainer settings to control the training process using PyTorch Lightning:

- `max_epochs`: Set the maximum number of training epochs.
- `precision`: Define the precision (e.g., 32-bit) for calculations.
- `val_check_interval`: Set the validation check interval during training.
- `gradient_clip_val`: Set the threshold for gradient clipping.
- `accumulate_grad_batches`: Define the number of batches to accumulate gradients.
- `accelerator`: Choose the accelerator type for training.
- `devices`: Specify the number of devices (GPUs) to utilize.

For more information, refer to the [PyTorch Lightning Trainer documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

### Model Settings

Configure the model settings based on your custom-designed architecture:

- `name`: Specify the name of your model.
- `backbone`: Customize the backbone architecture and its parameters.
- `neck`: Adjust neck architecture settings.
- `head`: Define head architecture parameters specific to your model.

### Dataset Settings

Adapt the dataset settings according to your custom dataset:

- `train_options`: Set the options for your training dataset, including paths, augmentation, and size.
- `valid_options`: Configure options for the validation dataset similarly to the training dataset.

### Dataloader Settings

Fine-tune the dataloader settings, leveraging PyTorch's DataLoader:

- `train_options`: Set the number of worker processes, shuffling, and batching for the training dataloader.
- `valid_options`: Configure similar settings for the validation dataloader.

### Optimizer Settings

Customize the optimizer settings using PyTorch's optimizers:

- `name`: Specify the optimizer name (e.g., "AdamW").
- `options`: Define optimizer options such as learning rate, weight decay, and betas.

### Learning Rate Scheduler Settings

Adapt the learning rate scheduler settings using PyTorch's scheduler:

- `name`: Specify the scheduler name (e.g., "MultiStepLRWarmUp").
- `options`: Configure scheduler-specific options like milestones and warm-up.

### Callbacks

Utilize PyTorch Lightning's callbacks to enhance training:

- Configure various callbacks like "ModelCheckpoint" and "LearningRateMonitor" to improve training efficiency and tracking.

### Logger

Leverage PyTorch Lightning's logger for monitoring training progress:

- Configure the "TensorBoardLogger" to visualize training metrics using TensorBoard.
