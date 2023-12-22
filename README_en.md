**[English](./README_en.md)** | [中文](./README.md)

# DocAligner

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligner/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligner?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

This project is a visual system focused on the localization of documents in the image. Our primary aim for this system is to provide predictions of the four corners of documents. This feature is critically important in applications dealing with fintech, banking, and the shared economy, offering a reduction in errors and computational requirements for various image processing and text analysis tasks.

<div align="center">
    <img src="./docs/title.jpg" width="800">
</div>

## Table of Contents
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Quick Start](#quick-start)
    - [Import Necessary Dependencies](#import-necessary-dependencies)
    - [ModelType](#modeltype)
    - [Backend](#backend)
    - [Create a DocAligner Instance](#create-a-docaligner-instance)
    - [Read and Process Images](#read-and-process-images)
    - [Output Results](#output-results)
        - [Draw Document Polygon](#draw-document-polygon)
        - [Get the Drawn numpy Image](#get-the-drawn-numpy-image)
        - [Extract the Flattened Document Image](#extract-the-flattened-document-image)
        - [Convert Document Information to JSON](#convert-document-information-to-json)
        - [Example](#example)
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
- [Convert to ONNX Format](#convert-to-onnx-format)

---

## Installation

Currently, we do not offer an installation package on Pypi. To use this project, you can directly clone it from Github and then install the necessary dependencies. Before proceeding with the installation, please ensure that you have [DocsaidKit](https://github.com/DocsaidLab/DocsaidKit) installed.

If you have already installed DocsaidKit, follow these steps:

1. Clone the project:

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. Enter the project directory:

   ```bash
   cd DocAligner
   ```

3. Create a packaging file:

   ```bash
   python setup.py bdist_wheel
   ```

4. Install the packaging file:

   ```bash
   pip install dist/docaligner-*-py3-none-any.whl
   ```

By following these steps, you should be able to successfully install DocAligner.

Once the installation is complete, you can start using the project.

---

## Quick Start

We provide a simple model inference interface, which includes pre-processing and post-processing logic.

First, you need to import the required dependencies and create a DocAligner class.

### Import Necessary Dependencies

```python
import docsaidkit as D
from docsaidkit import Backend
from docaligner import DocAligner, ModelType
```

### ModelType

`ModelType` is an enumeration type used to specify the model type for DocAligner. It includes the following options:

- `heatmap`: Uses a heatmap model for document alignment.
- `point`: Uses a point detection model for document alignment.

More model types may be added in the future, and we will update them here.

### Backend

`Backend` is an enumeration type used to specify the computational backend for DocAligner. It includes the following options:

- `cpu`: Uses the CPU for computations.
- `cuda`: Uses the GPU for computations (requires appropriate hardware support).

ONNXRuntime supports many backends, including CPU, CUDA, OpenCL, DirectX, TensorRT, etc. If you have other requirements, you can refer to [**ONNXRuntime Execution Providers**](https://onnxruntime.ai/docs/execution-providers/index.html) and modify it to the corresponding backend.

### Create a DocAligner Instance

```python
model = DocAligner(
    gpu_id=0,  # GPU ID, set to -1 if not using GPU
    backend=Backend.cpu,  # Choose the computational backend, can be Backend.cpu or Backend.cuda
    model_type=ModelType.point  # Choose the model type, can be ModelType.heatmap or ModelType.point
)
```

Note:

- Using the cuda backend requires not only appropriate hardware but also the installation of the corresponding CUDA drivers and toolkit. If CUDA is not installed in your system or the installed version is incorrect, the cuda backend will not be available.

- For issues related to ONNXRuntime installation dependencies, please refer to [ONNXRuntime Release Notes](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

### Read and Process Images

```python
# Read the image
img = D.imread('path/to/your/image.jpg')

# You can also use our test image
# img = D.imread('docs/run_test_card.jpg')

# Use the model for inference
result = model(img) # result is a Document type
```

### Output Results

The inference result you get is wrapped as a `Document` type, containing document polygons, OCR text information, etc.

The `Document` class offers various features to help process and analyze document images. Main features include:

1. **Document Polygon Processing**: Capable of identifying and manipulating document boundaries.
2. **OCR Text Recognition**: Supports recognizing text from images.
3. **Image Transformation**: Capable of transforming images based on document boundaries.

- Attributes
    - `image`: Stores the image of the document.
    - `doc_polygon`: The polygonal boundary of the document.
    - `ocr_texts`: The list of texts recognized by OCR.
    - `ocr_polygons`: The polygonal boundaries corresponding to `ocr_texts`.

- Methods
    - `gen_doc_flat_img()`: Transforms the document image according to its polygonal boundary.
    - `gen_doc_info_image()`: Generates an image marked with document boundaries and directions.
    - `gen_ocr_info_image()`: Generates an image showing OCR text and its boundaries.
    - `draw_doc()`: Saves the image marked with document boundaries to a specified path.
    - `draw_ocr()`: Saves the image showing OCR text and boundaries to a specified path.

In this module, we will not use OCR-related features, so we will only use the `image` and `doc_polygon` attributes. After obtaining the inference result, you can perform various post-processing operations.

#### Draw Document Polygon

```python
# Draw and save an image with the document polygon
result.draw_doc('path/to/save/folder', 'output_image.jpg')
```

Or if you don't specify a save path, it will be saved in the current directory with an automatically assigned timestamp.

```python
result.draw_doc()
```

#### Get the Drawn numpy Image

If you have other requirements, you can use the `gen_doc_info_image` method and then process it yourself.

```python
img = result.gen_doc_info_image()
```

#### Extract the Flattened Document Image

If you know the original size of the document, you can use the `gen_doc_flat_img` method to transform the document image into a rectangular image according to its polygonal boundary.

```python
H, W = 1080, 1920
flat_img = result.gen_doc_flat_img(image_size=(H, W))
```

For an unknown image type, you can also proceed without specifying the `image_size` parameter. In this case, the minimum rectangular image will be calculated based

 on the document polygon, and the length and width of the minimum rectangle will be set as `H` and `W`.

```python
flat_img = result.gen_doc_flat_img()
```

#### Convert Document Information to JSON

If you need to save document information to a JSON file, use the `be_jsonable` method.

When converting, consider excluding the image to save space, defaulting to `exclude_image=True`.

```python
doc_json = result.be_jsonable()
D.dump_json(doc_json)
```

#### Example

```python
import docsaidkit as D
from docaligner import DocAligner

model = DocAligner(D.Backend.cpu)
img = D.imread('docs/run_test_card.jpg')
result = model(img)

# You can draw the result by yourself.
output_img = D.draw_polygon(img, result.doc_polygon)
flat_img = result.gen_doc_flat_img(image_size=(480, 800))
D.imwrite(output_img)
D.imwrite(flat_img)

# Or you can draw the colorful image from `draw_doc` method.
# result.draw_doc()
```

<div align="center">
    <img src="./docs/flat_result.jpg" width="800">
</div>

---

## Benchmark

We utilized the [SmartDoc 2015](https://github.com/jchazalon/smartdoc15-ch1-dataset) dataset as our test dataset.

### Evaluation Protocol

We used the **Jaccard Index** as our measurement standard. This index summarizes the ability of different methods in accurately segmenting page outlines and penalizes those methods that fail to detect document objects in certain frames.

The evaluation process starts by using the size and coordinates of documents in each frame to perform a perspective transformation on the quadrilateral coordinates of the submitted method S and the ground truth G, resulting in the corrected quadrilaterals S0 and G0. Such transformation makes all evaluation metrics comparable within the document reference frame. For each frame f, the Jaccard Index (JI) is calculated, which is a measure of the extent of overlap between the corrected quadrilaterals, calculated as follows:

$$ JI(f) = \frac{\text{area}(G0 \cap S0)}{\text{area}(G0 \cup S0)} $$

where $` \text{area}(G0 \cap S0) `$ is defined as the intersection polygon of the detected quadrilateral and the ground truth quadrilateral, and $` \text{area}(G0 \cup S0) `$ is their union polygon. The overall score for each method will be the average of scores for all frames in the test dataset.

### Evaluation Results

The current model's performance has not yet reached the scores of state-of-the-art (SoTA) models, but it can already meet the needs of most application scenarios.

For instance, the `PointRec-256` model, with a current development scale of 6 MB and computational requirement of approximately 1.2 GFLOPs, can run on various devices, including mobile phones and embedded devices.

The `PointRec-512` model, while having the same model size, has double the input image resolution, thus quadrupling the computational requirement to about 5.0 GFLOPs.

We believe that training methods and dataset compositions are significant factors affecting model performance. Therefore, we will continue to update our models and provide more datasets to enhance the models' effectiveness.

| Models | bg01 | bg02 | bg03 | bg04 | bg05 | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PointRec-512 (Ours)** |  **0.9821** |  **0.9798** |  **0.9831** |  **0.9753** |  **0.9055** |  **0.9727**|
| **PointRec-256 (Ours)** |  **0.9681** |  **0.9597** |  **0.9720** |  **0.9597** |  **0.8864** |  **0.9571**|
| - | - | - | - | - | - | - |
| HU-PageScan | - | - | - | - | - | 0.9923 |
| Advanced Hough |  0.9886 |  0.9858 |  0.9896 |  0.9806 |  - |  0.9866 |
| LDRNet | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 | 0.9849 |
| Coarse-to-Fine |  0.9876 |  0.9839 |  0.9830 |  0.9843 |  0.9614 |  0.9823 |
| SEECS-NUST-2 |  0.9832 |  0.9724 |  0.9830 |  0.9695 |  0.9478 |  0.9743 |
| LDRE | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 | 0.9716 |
| SmartEngines |  0.9885 |  0.9833 |  0.9897 |  0.9785 |  0.6884 |  0.9548 |
| NetEase |  0.9624 |  0.9552 |  0.9621 |  0.9511 |  0.2218 |  0.8820 |
| RPPDI-UPE |  0.8274 |  0.9104 |  0.9697 |  0.3649 |  0.2162 |  0.7408 |
| SEECS-NUST |  0.8875 |  0.8264 |  0.7832 |  0.7811 |  0.0113 |  0.7393 |

---

## Training the Model

We do not offer the functionality for fine-tuning the model, but you can use our training module to produce a model yourself. Below, we provide a complete training process to help you start from scratch.

Broadly, you need to follow several steps:

1. **Prepare the Dataset**: Collect and organize data suitable for your needs.
2. **Set Up the Training Environment**: Configure the required hardware and software environment.
3. **Execute Training**: Train the model using your data.
4. **Evaluate the Model**: Test the model's performance and make adjustments.
5. **Convert to ONNX Format**: For better compatibility and performance, convert the model to ONNX format.
6. **Assess Quantization Needs**: Decide if quantization of the model is needed to optimize performance.
7. **Integrate and Package the Model**: Incorporate the ONNX model into your project.

Let's now break down the training process step-by-step.

---

Before we start, we understand that you may have the budget, but not enough time to customize your on-site environment.

Therefore, you can also contact us directly for consultation. Based on the difficulty of your project, we can arrange for engineers to conduct customized development.

A direct and specific example is:

If you need to capture a certain type of text in a specific angle and light source, and find that the model we provide does not perform well, you can contact us and provide some data you have collected. We can directly force the model to fit on your dataset. This method can significantly improve the model's performance, but it requires a lot of time and manpower, so we will provide a reasonable quote based on your needs.

Additionally, if you are not in a hurry, you can directly provide us with your text or scene dataset. We will include your dataset in our test datasets in future versions and offer better model performance in future releases. This method is completely free for you.

If you need more help, please contact us via email: **docsaidlab@gmail.com**

---

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
      cd DocAligner/data
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
    mode="val", # "train" or "val"
    train_ratio=0.2 # Using 20% of the data for training and 80% for validation.

# Only SmartDocDataset has the third return value,
# it's for validation and benchmarking.
img, poly, doc_type = ds[0]

# If set `mode="train"`, two return values will be returned.
# img, poly = ds[0]

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

First, ensure you have built the base image `docsaid_training_base_image` from `DocsaidKit`. If not, refer to `DocsaidKit` documentation. Then, use the following command to build the Docker image for DocAligner work:

```bash
# Build base image from docsaidkit at first
cd DocsaidKit
bash docker/build.bash

# Then build DocAligner image
cd DocAligner
bash docker/build.bash
```

Our default [Dockerfile](./docker/Dockerfile) is specifically designed for document alignment training. You may modify it as needed. Here is an explanation of the file, if you wish to make changes, you can refer to this:

- **Base Image**
    - `FROM docsaid_training_base_image:latest`
    - This line specifies the base image for the container, the latest version of `docsaid_training_base_image`. The base image is like a starting point for building your Docker container, containing a pre-configured operating system and some basic tools. You can find it in the `DocsaidKit` project.

- **Working Directory**
    - `WORKDIR /code`
    - The container's working directory is set to `/code`. This is a directory in the Docker container where your application and all commands will operate.

- **Environment Variable**
    - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
    - This defines an environment variable `ENTRYPOINT_SCRIPT` set to `/entrypoint.sh`. Environment variables store common configurations and can be accessed anywhere in the container.

- **Install gosu**
    - `RUN` command installs `gosu`, a lightweight tool that allows users to execute commands with a specific user identity, similar to `sudo` but more suited for Docker containers.
    - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` updates the package list, installs `gosu`, and then cleans up unnecessary files to reduce image size.

- **Create Entry Point Script**
    - A series of `RUN` commands create the entry point script `/entrypoint.sh`.
    - The script first checks if environment variables `USER_ID` and `GROUP_ID` are set. If so, it creates a new user and group with the same IDs and runs commands as that user.
    - Useful for handling file permission issues inside and outside the container, especially when the container needs to access files on the host machine.

- **Permission Assignment**
    - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` makes the entry point script executable.

- **Set Container Entry Point and Default Command**
    - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` and `CMD ["bash"]`
    - These commands specify the default command executed when the container starts. When the container launches, it runs the `/entrypoint.sh` script.

---

## Running Training (Based on Docker)

This section explains how to use the Docker image you've built for document alignment training.

First, examine the contents of the `train.bash` file:

```bash
#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocAligner.model import main_docaligner_train

if __name__ == '__main__':
    Fire(main_docaligner_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligner:/code/DocAligner \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python trainer.py --cfg_name $1
```

Here is an explanation of the file, if you wish to make changes, you can refer to this:

- **Create Training Script**
   - `cat > trainer.py <<EOF ... EOF`
   - This command creates a Python script `trainer.py`. This script imports necessary modules and functions and invokes `main_docalign_train` in the script's main section. It uses Google's Python Fire library for command-line argument parsing, making CLI generation easier.

- **Run Docker Container**
   - `docker run ... doc_align_train python trainer.py --cfg_name $1`
   - This command starts a Docker container and runs the `trainer.py` script inside it.
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`: These parameters pass the current user's user ID and group ID to the container to create a user with corresponding permissions inside.
   - `--gpus all`: Specifies that the container can use all GPUs.
   - `--shm-size=64g`: Sets the size of shared memory, useful in large-scale data processing.
   - `--ipc=host --net=host`: These settings allow the container to use the host's IPC namespace and network stack, helping with performance.
   - `--cpuset-cpus="0-31"`: Specifies which CPU cores the container should use.
   - `-v $PWD/DocAligner:/code/DocAligner` etc.: These are mount parameters, mapping directories from the host to the container for easy access to training data and scripts.
   - `--cfg_name $1`: This is an argument passed to `trainer.py`, specifying the name of the configuration file.

- **Dataset Path**
   - Note that `/data/Dataset` is the path for training data. Adjust `-v /data/Dataset:/data/Dataset` to match your dataset directory.

Finally, return to the `DocAligner` parent directory and execute the following command to start training:

```bash
# You should replace with your configuration file name
bash DocAligner/docker/train.bash lcnet150_fpn_d3_r256
```

- Note: For configuration file details, refer to [DocAligner/model/README.md](./model/README.md).

By following these steps, you can safely execute document alignment training tasks within a Docker container, leveraging Docker's isolated environment for consistency and reproducibility. This approach makes project deployment and scaling more convenient and flexible.

---

## Converting Model to ONNX Format (Based on Docker)

Here's a guide on how to convert your model to ONNX format using Docker.

First, let's look at the contents of the `to_onnx.bash` file:

```bash
#!/bin/bash

cat > torch2onnx.py <<EOF
from fire import Fire
from DocAligner.model import main_docaligner_torch2onnx

if __name__ == '__main__':
    Fire(main_docaligner_torch2onnx)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocAligner:/code/DocAligner \
    -v $PWD/torch2onnx.py:/code/torch2onnx.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python torch2onnx.py --cfg_name $1
```

You don't need to modify this file. Instead, you should make changes to the corresponding file: `model/to_onnx.py`.

During the training process, you might use many branches to supervise the model's training, but during the inference stage, you might only need one of those branches. Hence, we need to convert the model to ONNX format, retaining only the branch needed for inference.

For example:

```python
class WarpLC100FPN(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, img: torch.Tensor):
        return self.head(self.neck(self.backbone(img)))

```

In the example above, we extract the branch used for inference and wrap it into a new model `WarpLC100FPN`. Then, set the corresponding parameters in the yaml config:

```yaml
onnx:
  name: WarpLC100FPN
  input_shape:
    img:
      shape: [1, 3, 256, 256]
      dtype: float32
  input_names: ['img']
  output_names: ['output']
  dynamic_axes:
    img:
      '0': batch_size
    output:
      '0': batch_size
  options:
    opset_version: 16
    verbose: False
    do_constant_folding: True
```

This specifies the model's input size, input name, output name, and the ONNX version number.

We've already written the conversion part for you. After completing the modifications above, ensure that the `model/to_onnx.py` file is pointing to your model. Then, go up to the parent directory of `DocAligner` and run the following command to initiate the conversion:

```bash
bash DocAligner/docker/to_onnx.bash lcnet100_point_reg_bifpn # Replace this with your configuration file name
```

At this point, you will see a new ONNX model in the `DocAligner/model` directory. Move this model to the corresponding inference model directory in `docaligner/xxx` to perform inference.
