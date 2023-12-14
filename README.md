# DocAligned

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligned/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligned?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

This project is a visual system focused on the localization of documents in the image. Our primary aim for this system is to provide predictions of the four corners of documents. This feature is critically important in applications dealing with fintech, banking, and the shared economy, offering a reduction in errors and computational requirements for various image processing and text analysis tasks.

## Table of Contents
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Dataset Implementation](#dataset-implementation)
    - [1. SmartDoc 2015 Dataset](#1-smartdoc-2015-dataset)
    - [2. MIDV-500 Dataset](#2-midv-500-dataset)
    - [3. MIDV-2019 Dataset](#3-midv-2019-dataset)
    - [4. MIDV-2020 Dataset](#4-midv-2020-dataset)
    - [5. CORD v0 Dataset](#5-cord-v0-dataset)
    - [6. Synthetic Dataset](#6-synthetic-dataset)
    - [7. Image Augmentation](#7-image-augmentation)
- [Building the Training Environment](#building-the-training-environment)
- [Running Training (Based on Docker)](#running-training-based-on-docker)
- [Reference](#reference)

## Dataset

- **SmartDoc 2015**
    - [**SmartDoc 2015**](https://github.com/jchazalon/smartdoc15-ch1-dataset)
    - The Smartdoc 2015 - Challenge 1 dataset was originally created for the Smartdoc 2015 competition focusing on the evaluation of document image acquisition method using smartphones. The challenge 1, in particular, consisted in detecting and segmenting document regions in video frames extracted from the preview stream of a smartphone.

- **MIDV-500/MIDV-2019**
   - [**MIDV**](https://github.com/fcakyon/midv500)
   - MIDV-500 comprises 500 video clips of 50 different identity document types, including 17 ID cards, 14 passports, 13 driver's licenses, and 6 other identity documents from various countries. It is authentic and allows extensive research on various document analysis problems.
   - The MIDV-2019 dataset includes distorted and low-light images.

- **MIDV-2020:**
   - [**MIDV2020**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
   - MIDV-2020 is a dataset of 10 document types, including 1000 annotated video clips, 1000 scanned images, and 1000 unique photos of 1000 simulated identity documents, each with a unique text field value and a unique artificially generated face.

- **Indoor Scenes**
   - [**Indoor**](https://web.mit.edu/torralba/www/indoor.html)
   - This dataset contains 67 indoor categories, with a total of 15,620 images. The number of images varies by category, but each has at least 100 images. All images are in jpg format.

- **CORD v0**
   - [**CORD**](https://github.com/clovaai/cord)
   - Comprising thousands of Indonesian receipts, this dataset includes image text annotations for OCR and multi-layer semantic labels for parsing. It can be used for a variety of OCR and parsing tasks.

- **Docpool**
   - [**Dataset**](./data/docpool/)
   - We collected various text images from the internet for use in dynamic composite image technology as a training dataset.

## Dataset Preprocessing

1. **Install MIDV-500 Package:**

    ```bash
    pip install midv500
    ```

2. **Download Datasets:**

    - **MIDV-500/MIDV-2019:**
      After installation, execute `download_midv.py`.

      ```bash
      cd DocAligned/data
      python download_midv.py
      ```

    - **MIDV-2020:**
      Visit their respective links and follow the download instructions.

    - **SmartDoc 2015:**
      Visit their respective links and follow the download instructions.

    - **Indoor Scenes & CORD v0:**
      Visit their respective links and follow the download instructions.

3. **Build Dataset:**

    Place MIDV and CORD datasets in the same location, and set the `ROOT` variable in `build_dataset.py` to the directory where datasets are stored. Then, execute:

    ```bash
    python build_dataset.py
    ```

   This process will generate several `.json` files containing all dataset information, including image paths, labels, image sizes, etc.

## Dataset Implementation

We have implemented datasets corresponding to the several mentioned datasets for training in PyTorch. Please refer to [dataset.py](./model/dataset.py).

Below, we demonstrate how to load these datasets:

### 1. SmartDoc 2015 Dataset

```python
import docsaidkit as D
from model.dataset import SmartDocDataset

ds = SmartDocDataset(
    root="/data/Dataset" # Replace with your dataset directory
)

# Only SmartDocDataset has the third return value,
# it's for validation and benchmarking.
img, poly, _ = ds[0]

D.imwrite(D.draw_polygon(img, poly, thickness=5), 'smartdoc_test_img.jpg')
```

<div align="center">
    <img src="./docs/smartdoc_test_img.jpg" width="500">
</div>

### 2. MIDV-500 Dataset

```python
import docsaidkit as D
from model.dataset import MIDV500Dataset

ds = MIDV500Dataset(
    root="/data/Dataset" # Replace with your dataset directory
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'midv500_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv500_test_img.jpg" width="300">
</div>

### 3. MIDV-2019 Dataset

```python
import docsaidkit as D
from model.dataset import MIDV2019Dataset

ds = MIDV2019Dataset(
    root="/data/Dataset" # Replace with your dataset directory
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'midv2019_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv2019_test_img.jpg" width="300">
</div>

### 4. MIDV-2020 Dataset

```python
import docsaidkit as D
from model.dataset import MIDV2020Dataset

ds = MIDV2020Dataset(
    root="/data/Dataset" # Replace with your dataset directory
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=3), 'midv2020_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv2020_test_img.jpg" width="300">
</div>

### 5. CORD v0 Dataset

```python
import docsaidkit as D
from model.dataset import CordDataset

ds = CordDataset(
    root="/data/Dataset" # Replace with your dataset directory
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'cordv0_test_img.jpg')
```

<div align="center">
    <img src="./docs/cordv0_test_img.jpg" width="300">
</div>

### 6. Synthetic Dataset

Considering the limitations of the datasets, we use dynamic image synthesis technology.

In simple terms, we first collected a Docpool dataset, which includes images of various documents and IDs found on the internet. Then, we used the Indoor dataset as a background and synthesized the data from Docpool onto this background.

Furthermore, the MIDV-500/MIDV-2019/CORD datasets also have corresponding Polygon data. We also synthesize the images from Docpool onto these datasets to increase their diversity.

In short, just use it, and don't worry about the implementation details.

```python
import docsaidkit as D
from model.dataset import SyncDataset

ds = SyncDataset(
    root="/data/Dataset" # Replace with your dataset directory
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=2), 'sync_test_img.jpg')
```

<div align="center">
    <img src="./docs/sync_test_img.jpg" width="300">
</div>

### 7. Image Augmentation

Despite having collected some data, the diversity of these datasets is still insufficient. To increase the diversity, we use image augmentation techniques, which can simulate various conditions during image capture, such as occlusion, motion, rotation, blurring, noise, color changes, etc.

```python
import cv2
import numpy as np
import docsaidkit as D
import docsaidkit.torch as DT
import albumentations as A

class DefaultImageAug:

    def __init__(self, p=0.5):
        self.coarse_drop_aug = DT.CoarseDropout(
            max_holes=1, max_height=64, max_width=64, p=p)
        self.aug = A.Compose([
            DT.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=[-0.4, 0.2],
                border_mode=cv2.BORDER_CONSTANT),
            A.MotionBlur(),
            A.GaussNoise(),
            A.ColorJitter(),
            A.ChannelShuffle(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Perspective(),
            A.GaussianBlur(blur_limit=(7, 11), p=0.5),
        ], p=p, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> Any:
        img = self.coarse_drop_aug(image=image)['image']
        img, kps = self.aug(image=img, keypoints=keypoints).values()
        kps = D.order_points_clockwise(np.array(kps))
        return img, kps
```

- **CoarseDropout**
   - This augmentation technique randomly generates a rectangular area in the image and sets the pixel values in that area to 0. It can simulate occlusions in images, such as when text is obscured by other objects.

- **GaussianBlur**
    - This technique applies Gaussian blur to the image. It can simulate Gaussian blur during image capture and blur sharp edge features in synthetic images, making them look more like real images.

- **Others**
    - These augmentation techniques can simulate various conditions during image capture, such as motion, rotation, blurring

## Building the Training Environment

First, ensure you have built the base image `docsaid_training_base_image` from `DocsaidKit`. If not, refer to `DocsaidKit` documentation. Then, use the following command to build the Docker image for DocAligned work:

```bash
cd DocAligned
bash docker/build.bash
```

Our default [Dockerfile](./docker/Dockerfile) is specifically designed for document alignment training. You may modify it as needed. Here is an explanation of the file:

1. **Base Image**
    - `FROM docsaid_training_base_image:latest`
    - This line specifies the base image for the container, the latest version of `docsaid_training_base_image`. The base image is like a starting point for building your Docker container, containing a pre-configured operating system and some basic tools. You can find it in the `DocsaidKit` project.

2. **Working Directory**
    - `WORKDIR /code`
    - The container's working directory is set to `/code`. This is a directory in the Docker container where your application and all commands will operate.

3. **Environment Variable**
    - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
    - This defines an environment variable `ENTRYPOINT_SCRIPT` set to `/entrypoint.sh`. Environment variables store common configurations and can be accessed anywhere in the container.

4. **Install gosu**
    - `RUN` command installs `gosu`, a lightweight tool that allows users to execute commands with a specific user identity, similar to `sudo` but more suited for Docker containers.
    - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` updates the package list, installs `gosu`, and then cleans up unnecessary files to reduce image size.

5. **Create Entry Point Script**
    - A series of `RUN` commands create the entry point script `/entrypoint.sh`.
    - The script first checks if environment variables `USER_ID` and `GROUP_ID` are set. If so, it creates a new user and group with the same IDs and runs commands as that user.
    - Useful for handling file permission issues inside and outside the container, especially when the container needs to access files on the host machine.

6. **Permission Assignment**
    - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` makes the entry point script executable.

7. **Set Container Entry Point and Default Command**
    - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` and `CMD ["bash"]`
    - These commands specify the default command executed when the container starts. When the container launches, it runs the `/entrypoint.sh` script.

## Running Training (Based on Docker)

This section explains how to use the Docker image you've built for document alignment training.

First, examine the contents of the `train.bash` file:

```bash
#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocAligned.model import main_docalign_train

if __name__ == '__main__':
    Fire(main_docalign_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligned:/code/DocAligned \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python trainer.py --cfg_name $1
```

Explanation:

1. **Create Training Script**
   - `cat > trainer.py <<EOF ... EOF`
   - This command creates a Python script `trainer.py`. This script imports necessary modules and functions and invokes `main_docalign_train` in the script's main section. It uses Google's Python Fire library for command-line argument parsing, making CLI generation easier.

2. **Run Docker Container**
   - `docker run ... doc_align_train python trainer.py --cfg_name $1`
   - This command starts a Docker container and runs the `trainer.py` script inside it.
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`: These parameters pass the current user's user ID and group ID to the container to create a user with corresponding permissions inside.
   - `--gpus all`: Specifies that the container can use all GPUs.
   - `--shm-size=64g`: Sets the size of shared memory, useful in large-scale data processing.
   - `--ipc=host --net=host`: These settings allow the container to use the host's IPC namespace and network stack, helping with performance.
   - `--cpuset-cpus="0-31"`: Specifies which CPU cores the container should use.
   - `-v $PWD/DocAligned:/code/DocAligned` etc.: These are mount parameters, mapping directories from the host to the container for easy access to training data and scripts.
   - `--cfg_name $1`: This is an argument passed to `trainer.py`, specifying the name of the configuration file.

3. **Dataset Path**
   - Note that `/data/Dataset` is the path for training data. Adjust `-v /data/Dataset:/data/Dataset` to match your dataset directory.

Finally, return to the `DocAligned` parent directory and execute the following command to start training:

```bash
bash DocAligned/docker/train.bash LC150_BIFPN64_D3_PointReg_r256 # Replace with your configuration file name
```

- Note: For configuration file details, refer to [DocAligned/model/README.md](./model/README.md).

By following these steps, you can safely execute document alignment training tasks within a Docker container, leveraging Docker's isolated environment for consistency and reproducibility. This approach makes project deployment and scaling more convenient and flexible.


## Reference

- **Model Architecture**
    - [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/abs/2109.15099)
    - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070v7)
    - [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399)
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)

- **Dataset**
    - [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf)
    - [CORD: A Consolidated Receipt Dataset for Post-OCR Parsing](https://openreview.net/forum?id=SJl3z659UH)
    - [MIDV-500: A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream](https://arxiv.org/abs/1807.05786)
    - [MIDV-2019: Challenges of the modern mobile-based document OCR](https://arxiv.org/abs/1910.04009)
    - [MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis](https://arxiv.org/abs/2107.00396)

- **Research in the same field**
    - [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf)
    - [Real-time Document Localization in Natural Images by Recursive Application of a CNN](https://khurramjaved.com/RecursiveCNN.pdf)
    - [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532)
    - [LDRNet: Enabling Real-time Document Localization on Mobile Devices](https://arxiv.org/abs/2206.02136)
