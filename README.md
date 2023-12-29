**[English](./README.md)** | [中文](./README_cn.md)

# DocAligner

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligner/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligner?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

<div align="center">
    <img src="./docs/title.jpg" width="800">
</div>

This project aims to develop a visual system specifically for the precise localization of documents within images. Our primary goal is to accurately predict the positions of the four corners of a document. This technology is particularly applicable in industries such as fintech, banking, and the sharing economy, as it significantly reduces the error rate and computational demands of image processing and text analysis tasks.

The core functionality of this system is known as "Document Localization." Our models are specifically designed to identify documents in images and flatten them for subsequent text recognition or other processing tasks. We offer two different models: the "Heatmap Model" and the "Point Regression Model," each with its characteristics and suitable applications, which will be detailed in subsequent sections.

Technically, we have chosen PyTorch as our training framework and ONNXRuntime for model inference, enabling efficient operation of our models on both CPUs and GPUs. Additionally, we support converting our models into the ONNX format for convenient deployment on various platforms. For scenarios requiring quantization, we provide a static quantization model function based on the ONNXRuntime API.

Our models achieve near state-of-the-art (SoTA) performance and demonstrate real-time inference speeds in practical applications, meeting the needs of most usage scenarios.

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Quick Start](#quick-start)
- [Benchmark](#benchmark)
- [Before We start Training](#before-we-start-training)
- [Training the Model](#training-the-model)
- [Model Architecture Design](#model-architecture-design)
- [Dataset](#dataset)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Dataset Implementation](#dataset-implementation)
- [Building the Training Environment](#building-the-training-environment)
- [Running Training (Based on Docker)](#running-training-based-on-docker)
- [Convert to ONNX Format](#convert-to-onnx-format)
- [Dataset Submission](#dataset-submission)
- [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)
- [Citation](#citation)

---

## Quick Start

### Installation

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

### Import Necessary Dependencies

We provide a simple model inference interface, which includes pre-processing and post-processing logic.

First, you need to import the required dependencies and create a DocAligner class.

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

### Performance Evaluation

We have provided a simple evaluation module that can be used to assess the model's performance on the SmartDoc2015 dataset.

Please ensure that you have downloaded the `SmartDoc2015` dataset and placed it in the `/data/Dataset` directory before running the evaluation. If the dataset is not in this directory, you can still proceed, but you will need to modify the path `-v /data/Dataset:/data/Dataset` in the `DocAligner/docker/benchmark.bash` script accordingly.

```bash
# The input content in order is:
# Target bash for execution, dataset name, model type, and model name.
bash DocAligner/docker/benchmark.bash smartdoc heatmap lcnet050
```

### Evaluation Results

<div align="center">

| Models | bg01 | bg02 | bg03 | bg04 | bg05 | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HeatmapRec-LC100-256 (Ours) |  0.9908 |  0.9877 |  0.9905 |  0.9894 |  0.9854 |  0.9892 |
| HeatmapRec-LC050-256 (Ours) |  0.9847 |  0.9822 |  0.9865 |  0.9811 |  0.9722 |  0.9826 |
| - | - | - | - | - | - | - |
| HU-PageScan [1] | - | - | - | - | - | 0.9923 |
| Advanced Hough [2] |  0.9886 |  0.9858 |  0.9896 |  0.9806 |  - |  0.9866 |
| LDRNet [4] | 0.9877 | 0.9838 | 0.9862 | 0.9802 | 0.9858 | 0.9849 |
| Coarse-to-Fine [3] |  0.9876 |  0.9839 |  0.9830 |  0.9843 |  0.9614 |  0.9823 |
| SEECS-NUST-2 [3] |  0.9832 |  0.9724 |  0.9830 |  0.9695 |  0.9478 |  0.9743 |
| LDRE [5] | 0.9869 | 0.9775 | 0.9889 | 0.9837 | 0.8613 | 0.9716 |
| SmartEngines [5] |  0.9885 |  0.9833 |  0.9897 |  0.9785 |  0.6884 |  0.9548 |
| NetEase [5] |  0.9624 |  0.9552 |  0.9621 |  0.9511 |  0.2218 |  0.8820 |
| RPPDI-UPE [5] |  0.8274 |  0.9104 |  0.9697 |  0.3649 |  0.2162 |  0.7408 |
| SEECS-NUST [5] |  0.8875 |  0.8264 |  0.7832 |  0.7811 |  0.0113 |  0.7393 |

</div>

1. **HU-PageScan** is a pixel classification-based segmentation model known for its good performance. However, its large size and computational demand, along with architectural limitations, reduce its resistance to partially occluded patterns, such as scenarios where fingers are holding the edges of a page. This makes it less suitable for practical applications.
    - Paper: [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532) (Feb 2021)
    - Github: [HU-PageScan](https://github.com/ricardobnjunior/HU-PageScan)

2. **Advanced Hough** is a CV-Based model with notable performance. However, like all CV-Based models, it has certain drawbacks, such as sensitivity to lighting and angles.
    - Paper: [Advanced Hough-based method for on-device document localization](https://www.computeroptics.ru/KO/PDF/KO45-5/450509.pdf) (June 2021)
    - Github:  [hough_document_localization](https://github.com/SmartEngines/hough_document_localization)

3. **Coarse-to-Fine** and **SEECS-NUST-2** are deep learning-based models that employ recursive optimization strategies. They perform well but are relatively slow.
    - Paper: [Real-time Document Localization in Natural Images by Recursive Application of a CNN](https://khurramjaved.com/RecursiveCNN.pdf) (Nov 2017)
    - Paper: [Coarse-to-fine document localization in natural scene image with regional attention and recursive corner refinement](https://sci-hub.et-fine.com/10.1007/s10032-019-00341-0) (July 2019)
    - Github:  [Recursive-CNNs](https://github.com/KhurramJaved96/Recursive-CNNs)

4. **LDRNet** is a deep learning-based model. Testing with the model provided by its creators showed that it is entirely fitted to the SmartDoc 2015 dataset and lacks generalizability to other scenarios. Attempts to incorporate additional data for training did not yield satisfactory results, possibly due to the architecture's insufficient capability for feature fusion.
    - Paper: [LDRNet: Enabling Real-time Document Localization on Mobile Devices](https://arxiv.org/abs/2206.02136) (June 2022)
    - Github:  [LDRNet](https://github.com/niuwagege/LDRNet)

5. Models such as **LDRE**, **SmartEngines**, **NetEase**, **RPPDI-UPE**, and **SEECS-NUST** are all based on CV-Based approaches.
    - Paper: [ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)](https://marcalr.github.io/pdfs/ICDAR15e.pdf) (Nov 2015)
    - Github:  [smartdoc15-ch1-dataset](https://github.com/jchazalon/smartdoc15-ch1-dataset)

---

### Analysis of Results

- Although our model can achieve near state-of-the-art (SoTA) scores, real-world scenarios are much more complex than this dataset. Therefore, it's not necessary to focus too much on these scores. Our aim is to demonstrate the effectiveness of our model.

- We have endeavored to minimize the size and computational requirements of our model. However, in our experiments, we found that the model's zero-shot capabilities are limited. This means that for new scenes, the model requires fine-tuning to achieve optimal performance.

- Through our testing, we have found that the 'heatmap regression model' is significantly more stable than the 'point regression model'. Therefore, we still recommend using the heatmap model.

- However, we cannot disregard the advantages of the 'point regression model', which include, but are not limited to: the ability to predict corner points outside the scope of the image; and a fast and simple post-processing procedure. Hence, we will continue to optimize the 'point regression model' to enhance its performance.

- HeatmapRec-LC100-256:
    - parameters: about 1.2 million
    - FP32 model file size: about 4.9 MB
    - computational: about 1.6 FLOPs(G)。

- HeatmapRec-LC050-256:
    - parameters: about 0.42 million
    - FP32 model file size: about 1.7 MB
    - computational: about 1.2 FLOPs(G)。


---

## Before We start Training

Based on the models we provide, we believe we can address most application scenarios. However, we recognize that some situations may require better model performance, necessitating the collection of specific datasets for model fine-tuning. We understand that you might have the budget but not the time to customize your on-site environment. Therefore, you can contact us directly for consultation. Depending on the complexity of your project, we can arrange for engineers to develop custom solutions for you.

Here's a specific example:

Suppose you need to extract text from a specific angle and lighting condition, and you find that our provided model does not perform well. In this case, you can contact us and provide some of the data you've collected. We can then tailor the model to fit your dataset directly. This approach can significantly improve the model's performance, but it requires a considerable amount of time and manpower. Therefore, we will provide a reasonable quote based on your needs.

Alternatively, if you are not in a hurry, **you can directly provide us with your dataset**. We will include your dataset in our test datasets in a future version (without a set timeline) to enhance the model's performance in subsequent releases. This option is entirely free for you.

- **Please note: We will never open-source the data you provide unless you specifically request it. Normally, the data is only used for model updates.**

We are delighted if you choose the second option, as it helps us improve our model, benefiting a broader audience.

For instructions on how to submit your dataset, please see: [**Dataset Submitting**](#dataset-submission).

Contact us at: **docsaidlab@gmail.com**

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

## Model Architecture Design

### Heatmap Regression Model

<div align="center">
    <img src="./docs/hmap_model_arch.jpg" width="800">
</div>

- **Backbone: LCNet**

    The Backbone serves as the main body of the model, responsible for extracting features from the input data.

    In this model, LCNet is utilized, a lightweight convolutional neural network that is especially effective for efficient feature extraction in environments with limited computational resources. We expect LCNet to extract sufficient feature information from the input data, preparing the groundwork for subsequent heatmap regression.

- **Neck: BiFPN**

    The Neck is used to enhance the features flowing from the Backbone.

    BiFPN (Bidirectional Feature Pyramid Network) enhances the feature representation by facilitating bidirectional flow of contextual information. We anticipate that BiFPN will produce a series of scale-rich and semantically strong feature maps. These feature maps are particularly effective for capturing objects at various scales and are expected to positively impact the final prediction accuracy.

- **Head: Heatmap Regression**

    The Head is the final stage of the model, specifically designed to make final predictions based on the features extracted and enhanced earlier.

    In this model, heatmap regression technique is employed. It is a common method in object detection and pose estimation, capable of accurately predicting object locations. Heatmap regression will generate a heatmap representation of objects, indicating the likelihood of object presence at different locations. By analyzing these heatmaps, the model can accurately predict the position and posture of objects.

- **Loss: Adaptive Wing Loss**

    Loss functions are crucial in model training as they compute the discrepancy between the model's predictions and actual labels.

    In this model, we have implemented Adaptive Wing Loss, a specialized loss function designed for facial landmark detection. This innovative approach to loss functions in heatmap regression is particularly suited for face alignment challenges. Its core concept involves adjusting the loss function's shape based on different types of pixels in the actual heatmap, imposing greater penalties on foreground pixels (i.e., pixels near facial feature points) and lesser penalties on background pixels.

    Here, we approach the problem of document corner prediction as akin to facial landmark detection, utilizing a loss function specifically designed for this purpose. We believe this method effectively addresses issues in document corner detection and performs well in various scenarios.

    **Reference: [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399)**

    Beyond the corner loss, we also employed multiple auxiliary losses, including:

    - **Edge Loss:** This loss function, used for edge detection, also employs heatmap regression and utilizes AWing Loss.
    - **Classification Loss:** This is a loss function for classification tasks, predicting the presence of documents in images, using BCE Loss.

### Point Regression Model

<div align="center">
    <img src="./docs/point_model_arch.jpg" width="800">
</div>

- **Backbone: LCNet**

    The backbone used here is the same as the one in the heatmap regression model; we have employed LCNet for feature extraction.

    Why not abandon the CNN architecture in favor of a full Transformer structure? The reason lies in limitations of parameter and computation quantities. Our experiments showed that when the number of parameters is too low, the Transformer cannot demonstrate its strengths, leading us to choose LCNet as our backbone.

- **Neck: Cross-Attention**

    The neck part enhances features flowing from the Backbone.

    In this model, we utilized a Cross-Attention mechanism, a common technique in Transformers, to capture relationships between different features, applying them to enhance feature representation. We expect Cross-Attention to aid the model in understanding spatial relationships between points in an image, thereby increasing prediction accuracy. In addition to Cross-Attention, positional encodings are also used. These encodings help the model comprehend the spatial positions of points in the image, further improving accuracy.

    Considering the characteristics of point regression, which relies heavily on low-level features for precise pixel positioning, we start from deeper features and sequentially query shallower ones (from 1/32 to 1/16 to 1/8 to 1/4 to 1/2). Such a design allows the model to locate documents in features of different scales. We believe this querying method effectively enhances model accuracy.

- **Head: Point Regression**

    A simple linear layer is used as the Head, transforming features into point coordinates. Our aim is for the model to rely more on the expressive capability of Cross-Attention features, rather than a complex head structure.

- **Loss: Smooth L1 Loss**

    In our model, we have chosen to use the Smooth L1 Loss as our loss function. This is a commonly used loss function in regression tasks, particularly effective for handling outliers. Compared to the traditional L1 Loss, the Smooth L1 Loss is more robust when the difference between predicted and actual values is significant, thereby reducing the impact of outliers on model training. Additionally, to minimize the amplified error in point regression, we increased the weight of point prediction to 1000. Our experiments have shown that such a design effectively improves model accuracy.

    Apart from the corner loss, we have also employed other losses, including:

    - **Classification Loss:** This is a loss function for classification, used to predict whether a document is present in an image, utilizing BCE Loss.

    It is important to note that here, the classification loss is not just an auxiliary loss but one of the primary losses. Given the inherent limitations of corner prediction, which can predict corners even in the absence of a target, we rely on the classification head during deployment to inform us whether there is a target object present.

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

DIR = D.get_curdir(__file__)

ds = D.load_json(DIR.parent / 'data' / 'indoor_dataset.json')

bg_dataset = []
for data in D.Tqdm(ds):
    img_path = D.Path('/data/Dataset') / data['img_path']
    if D.imread(img_path) is None:
        continue
    bg_dataset.append(img_path)


class DefaultImageAug:

    def __init__(self, p=0.5):
        self.coarse_drop_aug = DT.CoarseDropout(
            max_holes=1,
            min_height=24,
            max_height=48,
            min_width=24,
            max_width=48,
            mask_fill_value=255,
            p=p
        )
        self.aug = A.Compose([

            DT.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=[-0.2, 0]
            ),

            A.OneOf([
                A.Spatter(mode='mud'),
                A.GaussNoise(),
                A.ISONoise(),
                A.MotionBlur(),
                A.Defocus(),
                A.GaussianBlur(blur_limit=(3, 11), p=0.5),
            ], p=p),

            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
            ], p=p),

            A.OneOf([
                A.ColorJitter(),
                A.ChannelShuffle(),
                A.ChannelDropout(),
                A.RGBShift(),
            ])

        ], p=p, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> Any:
        mask = np.zeros_like(image)
        img, mask = self.coarse_drop_aug(image=image, mask=mask).values()
        background = bg_dataset[np.random.randint(len(bg_dataset))]
        background = D.imread(background)
        background = D.imresize(background, (image.shape[0], image.shape[1]))
        if mask.sum() > 0:
            img[mask > 0] = background[mask > 0]
        img, kps = self.aug(image=img, keypoints=keypoints).values()
        kps = D.order_points_clockwise(np.array(kps))
        return img, kps
```

- **CoarseDropout**
   - This augmentation technique randomly generates a rectangular area in the image and sets the random background in that area. It can simulate occlusions in images, such as when text is obscured by other objects.

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

---

## Dataset Submission

We greatly appreciate your willingness to provide a dataset for integration and testing. Here's how to format your submission:

**Example Dataset Format**:

<div align="center">
    <img src="./docs/example_dataset.jpg" width="500">
</div>

As you can see, you should have a dataset containing the images you've collected, along with a `gt.json` file in the same directory. This file should include labels for each image.

**Required Label Format**:

1. Image Relative Path
2. Polygon boundaries of the document's 'four corners' in the image

    ```json
    [
        {
            "file_path": "path/to/your/image.jpg",
            "polygon": [
                [
                    [0, 0],
                    [0, 1080],
                    [1920, 1080],
                    [1920, 0]
                ]
            ]
        }
    ]
    ```

Please note that while the above data format and naming conventions are not strict, they should generally include the image path and polygon boundaries. To facilitate our testing, please try to follow the format as closely as possible.

We recommend using [LabelMe](https://github.com/labelmeai/labelme), an open-source labeling tool that can help you quickly label images and export them as JSON files.

We suggest uploading your data to Google Drive and sharing the link with us via [email](docsaidlab@gmail.com). We will promptly test and integrate your data upon receipt. If your data does not meet our requirements, we will notify you as soon as possible.

**Possible Reasons for Non-compliance**:
- **Insufficient Data Accuracy**: If some images in your dataset are inaccurately labeled or have incorrect labels.
- **Unclear Labeling Objectives**: Our focus is on locating the four corners of documents in images. If your data contains more than one target or more than four corners, it cannot be used.
- **Too Small Targets**: If the objects in your dataset are too small, you might need to reconsider your algorithm choice, as our model is not suited for processing small objects and does not align with our goals for ease of post-processing.
- **Overly Refined Dataset Scale**: Even if you provide only a few dozen images, we will accept them. However, using such data to fit the model could lead to overfitting, so we recommend increasing the dataset size to avoid this issue.

---

## Frequently Asked Questions (FAQs)

1. **Is the order of the four corners important?**
   - No, it's not important. Our training process automatically sorts these corners.

2. **What are the requirements for the label format?**
   - The format is not strictly defined; it only needs to include the image path and the polygon boundaries. However, for ease of testing, we recommend adhering to a standard format as much as possible.

3. **How important is the file name?**
   - The file name is not a primary concern, as long as it correctly links to the corresponding image.

4. **Any recommendations for the image format?**
   - We suggest using the jpg format to save space.

5. **How does the accuracy of labels affect model training?**
   - The accuracy of the labels is extremely important. Inaccurate labels will directly impact the effectiveness of the model training.

6. **Is the type of object in the labels important?**
   - Yes, it's very important.
   - The object must be a document, and there should only be one object per image.

7. **How does the size of the object affect model training?**
   - The size of the object is important. Our model is not suitable for processing small objects, as this affects the efficiency of subsequent processing.

8. **How is a 'small object' defined?**
   - For an image with a resolution of 1920x1080, an object smaller than 32 x 32 pixels is considered a small object. The specific calculation formula is `min(img_w, img_h) / 32`.

For further assistance, please contact us via email at **docsaidlab@gmail.com**

---

## Citation

We are grateful to all those who have paved the way before us, their work has been immensely helpful to our research.

If you find our work helpful, please cite the following related papers:

```bibtex
@misc{yuan2023docaligner,
  author = {Ze Yuan},
  title = {DocAligner},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DocsaidLab/DocAligner}}
}

@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}

@inproceedings{quattoni2009recognizing,
  title={Recognizing indoor scenes},
  author={Quattoni, Ariadna and Torralba, Antonio},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={413--420},
  year={2009},
  organization={IEEE}
}

@inproceedings{park2019cord,
  title={CORD: a consolidated receipt dataset for post-OCR parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk},
  booktitle={Workshop on Document Intelligence at NeurIPS 2019},
  year={2019}
}

@article{arlazarov2019midv,
  title={MIDV-500: a dataset for identity document analysis and recognition on mobile devices in video stream},
  author={Arlazarov, Vladimir Viktorovich and Bulatov, Konstantin Bulatovich and Chernov, Timofey Sergeevich and Arlazarov, Vladimir Lvovich},
  journal={Компьютерная оптика},
  volume={43},
  number={5},
  pages={818--824},
  year={2019},
  publisher={Федеральное государственное автономное образовательное учреждение высшего~…}
}

@inproceedings{bulatov2020midv,
  title={MIDV-2019: challenges of the modern mobile-based document OCR},
  author={Bulatov, Konstantin and Matalov, Daniil and Arlazarov, Vladimir V},
  booktitle={Twelfth International Conference on Machine Vision (ICMV 2019)},
  volume={11433},
  pages={717--722},
  year={2020},
  organization={SPIE}
}

@article{bulatovich2022midv,
  title={MIDV-2020: a comprehensive benchmark dataset for identity document analysis},
  author={Bulatovich, Bulatov Konstantin and Vladimirovna, Emelianova Ekaterina and Vyacheslavovich, Tropin Daniil and Sergeevna, Skoryukina Natalya and Sergeevna, Chernyshova Yulia and Zuheng, Ming and Jean-Christophe, Burie and Muzzamil, Luqman Muhammad},
  journal={Компьютерная оптика},
  volume={46},
  number={2},
  pages={252--270},
  year={2022},
  publisher={Федеральное государственное автономное образовательное учреждение высшего~…}
}

@inproceedings{Wang_2019_ICCV,
author = {Wang, Xinyao and Bo, Liefeng and Fuxin, Li},
title = {Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}

@inproceedings{tan2020efficientdet,
  title={Efficientdet: Scalable and efficient object detection},
  author={Tan, Mingxing and Pang, Ruoming and Le, Quoc V},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10781--10790},
  year={2020}
}

@article{cui2021pp,
  title={PP-LCNet: A lightweight CPU convolutional neural network},
  author={Cui, Cheng and Gao, Tingquan and Wei, Shengyu and Du, Yuning and Guo, Ruoyu and Dong, Shuilong and Lu, Bin and Zhou, Ying and Lv, Xueying and Liu, Qiwen and others},
  journal={arXiv preprint arXiv:2109.15099},
  year={2021}
}

@inproceedings{burie2015icdar2015,
  title={ICDAR2015 competition on smartphone document capture and OCR (SmartDoc)},
  author={Burie, Jean-Christophe and Chazalon, Joseph and Coustaty, Micka{\"e}l and Eskenazi, S{\'e}bastien and Luqman, Muhammad Muzzamil and Mehri, Maroua and Nayef, Nibal and Ogier, Jean-Marc and Prum, Sophea and Rusi{\~n}ol, Mar{\c{c}}al},
  booktitle={2015 13th International Conference on Document Analysis and Recognition (ICDAR)},
  pages={1161--1165},
  year={2015},
  organization={IEEE}
}
```
