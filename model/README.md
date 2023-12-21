# Training Overview

我們有一個配置參數文件的方式，可以自定義訓練過程。此文件位於 `config/***.yaml`。您可以使用此文件來調整各種設置，以匹配您的特定模型架構、數據集和訓練要求。

## 環境要求

請確保您的環境符合以下要求：

- Python 3.8 或更高版本

## 執行訓練

執行以下命令開始訓練過程：

```bash
python trainer.py --cfg_name lcnet_100_fuse_all_edge
```

此命令將使用 `trainer.py` 腳本以及指定的配置名稱 `lcnet_100_fuse_all_edge` 來啟動模型訓練。確保您的當前工作目錄是包含腳本的目錄，當運行此命令時。

## 配置自定義

自定義訓練過程涉及調整各種設置以匹配您的特定模型架構、數據集和訓練要求。以下各節提供了如何使用 `config/***.yaml` 中提供的配置來自定義訓練的各個方面的詳細信息：

### 訓練器設置

調整訓練器設置以控制使用 PyTorch Lightning 的訓練過程：

- `max_epochs`：設置訓練的最大輪次（epoch）數。
- `precision`：定義計算的精度（例如，32位）。
- `val_check_interval`：設置訓練期間的驗證檢查間隔。
- `gradient_clip_val`：設置梯度剪裁的閾值。
- `accumulate_grad_batches`：定義累積梯度的批次數量。
- `accelerator`：選擇用於訓練的加速器類型。
- `devices`：指定使用的設備（GPU）數量。

欲了解更多資訊，請參閱 [PyTorch Lightning 訓練器文檔](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)。

### 模型設置

根據您自定義的架構配置模型設置：

- `name`：指定模型的名稱。
- `backbone`：自定義主幹架構及其參數。
- `neck`：調整頸部架構的設置。
- `head`：定義適用於您模型的頭部架構參數。

### Dataset

根據您的自定義數據集調整數據集設置：

- `train_options`：設置訓練數據集的選項，包括路徑、增強和大小。
- `valid_options`：類似於訓練數據集，配置驗證數據集的選項。

### DataLoader

利用 PyTorch 的 DataLoader 精調數據加載器設置：

- `train_options`：設置

訓練數據加載器的工作進程數、隨機排序和批次處理。
- `valid_options`：為驗證數據加載器配置類似設置。

### Optimizer

使用 PyTorch 的優化器自定義優化器設置：

- `name`：指定優化器名稱（例如：AdamW）。
- `options`：定義優化器選項，如學習率、權重衰減和betas。

### Learning Rate Scheduler

使用 PyTorch 的調度器調整學習率調度器設置：

- `name`：指定調度器名稱（例如：MultiStepLRWarmUp）。
- `options`：配置調度器特定的選項。

### Callbacks

利用 PyTorch Lightning 的回調來增強訓練：

- 配置各種回調，如：`ModelCheckpoint` 和 `LearningRateMonitor`，以提高訓練效率和跟踪。

### Logger

利用 PyTorch Lightning 的日誌記錄器監控訓練進展：

- 配置 `TensorBoardLogger`，使用 TensorBoard 可視化訓練指標。
