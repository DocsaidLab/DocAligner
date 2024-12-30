**[English](./README.md)** | [中文](./README_tw.md)

# DocAligner

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligner/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligner?color=ffa"></a>
    <a href="https://pypi.org/project/docaligner_docsaid/"><img src="https://img.shields.io/pypi/v/docaligner_docsaid.svg"></a>
</p>

## Introduction

<div align="center">
    <img src="https://github.com/DocsaidLab/DocAligner/blob/main/docs/title.jpg?raw=true" width="800">
</div>

This model is specifically designed to recognize documents in images and flatten them for subsequent text recognition or other processing.

We chose PyTorch as the training framework and converted the model to ONNX format for inference to facilitate deployment on various platforms. We use ONNXRuntime for model inference, enabling efficient operation on both CPU and GPU.

## Technical Documentation

Due to the extensive usage instructions and setup explanations for this project, we have only summarized the "Model Design" section here.

For installation and usage instructions, please refer to the [**DocAligner Documents**](https://docsaid.org/en/docs/docaligner/).

## Installation

### via PyPI

1. Install the package from PyPI:

   ```bash
   pip install docaligner-docsaid
   ```

2. Verify the installation:

   ```bash
   python -c "import docaligner; print(docaligner.__version__)"
   ```

3. If the version number is displayed, the installation was successful.

### via Git Clone

1. Clone this repository:

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the wheel file:

   ```bash
   cd DocAligner
   python setup.py bdist_wheel
   ```

4. Install the built wheel file:

   ```bash
   pip install dist/docaligner_docsaid-*-py3-none-any.whl
   ```

## Inferencing

We provide a simple inferencing script for model testing.

```python
import cv2
from skimage import io
from docaligner import DocAligner

model = DocAligner()
img = io.imread('https://github.com/DocsaidLab/DocAligner/blob/main/docs/run_test_card.jpg?raw=true')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

result = model(img)
# print(result)
#    [[ 48.151894 223.47687 ]
#    [387.1344   198.09961 ]
#    [423.0362   345.51334 ]
#    [ 40.148613 361.38782 ]]
# The result is a list of four corner points of the document.
```

## Model Testing

We provide a web-based model testing tool for preliminary model evaluation.

- [**DocAligner Web Demo**](https://docsaid.org/en/playground/docaligner-demo/)

If you have custom requirements, please feel free to contact us:

- **docsaidlab@gmail.com**

## Model Design

We reviewed previous research literature and initially considered a point regression model.

### Point Regression Model

![arch_1.jpg](https://github.com/DocsaidLab/DocAligner/blob/main/docs/point_model_arch.jpg?raw=true)

The point regression model is our earliest version, with its basic architecture divided into four parts:

1. **Feature Extraction**

   ![pp-lcnet.jpg](https://github.com/DocsaidLab/DocAligner/blob/main/docs/lcnet_arch.jpg?raw=true)

   This part is mainly used to convert images into vectors, using [**PP-LCNet**](https://arxiv.org/abs/2109.15099) as the feature extractor.

   The input image is a 128 x 128 RGB image, and after feature extraction, it outputs a 256-dimensional vector.

2. **Cross-Attention**

   In this model, the Neck part is used to enhance the features flowing from the Backbone.

   We used the Cross-Attention mechanism, a common mechanism in Transformers, to capture relationships between different features and apply these relationships to feature enhancement. We expect Cross-Attention to help the model understand the spatial relationships between different points in the image, thereby improving prediction accuracy. Besides Cross-Attention, we also used positional encodings, which help the model understand the spatial positions of points in the image, further improving prediction accuracy.

   Considering the characteristics of point regression, precise pixel localization relies heavily on low-level features. Therefore, we start from deep features and sequentially query shallower features (1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2). This design allows the model to find the document's location in features at different scales. We believe this querying method effectively improves the model's accuracy.

3. **Point Regression**

   In the design of the prediction head, we only use a simple linear layer as the Head to convert features into point coordinates. We hope the model can rely more on the expressive ability of Cross-Attention features rather than a complex head structure.

4. **Smooth L1 Loss**

   In our model, we chose Smooth L1 Loss as the loss function, which is commonly used in regression tasks, especially suitable for handling outliers.

   Compared with L1 Loss, Smooth L1 Loss is more robust when the difference between the predicted and actual values is large, reducing the impact of outliers on model training. Additionally, to reduce the amplification error in point regression, we increased the weight of point prediction "to 1000". Our experiments showed that this design effectively improves the model's accuracy.

   Besides the corner loss, we also used other losses, including:

   - Classification Loss: This is a loss function used for classification to predict whether a document exists in the image, using BCE Loss.

   Note that the classification loss here is not just an auxiliary loss but one of the main losses. Due to the limitations of corner point prediction, it will still predict corner points even when there is no object. Therefore, in the deployment phase, we need the classification head to tell us whether there is an object.

### Catastrophic Failure

In the "point regression model" architecture, we encountered a severe "amplification error" problem.

The root of this problem is that during the model training process, we need to downscale the original image to 128 x 128 or 256 x 256. This downscaling process results in the loss of detail information in the original image, making it difficult for the model to accurately locate the document's corners during prediction.

**Specifically, the model finds the corners based on the downscaled image.**

Then, we must enlarge these corner points to the original image size to find the corner positions in the original image.

This enlargement process causes the corner positions to shift by about 5 to 10 pixels, preventing the model from accurately predicting the document's location.

- **Note:** You can imagine that in the original image, the image within a 10-pixel radius of the target corner is reduced to 1 pixel during prediction. Then the model makes predictions, and in the enlargement process, the corner positions shift.

### How Do Others Solve This?

After encountering this problem, we consciously looked for how others solved it.

We found that in the field of Document Localization, many researchers addressed this issue by:

1. **Using Large Images for Prediction**

   This approach ensures that the model can accurately locate the document's corners during prediction.

   But it is very slow, extremely slow.

2. **Introducing Anchors and Offsets**

   The anchor-based approach can refer to the object detection field, where some prior knowledge is needed to define anchor sizes. However, documents can appear at any angle and deformation in the image, and the design of anchors limits the model's detection capability within a certain range.

   Essentially, the pros and cons of anchor structures you know can be rewritten here.

   Documents in the real world have arbitrary aspect ratios, making anchor design unsuitable.

3. **Directly Fitting Evaluation Datasets**

   Early papers on this topic often designed algorithms directly for SmartDoc 2015 rather than creating a general model.

   In recent years, papers split the SmartDoc 2015 dataset into training and testing sets by themselves to improve scores.

   So you see many architectures scoring well on benchmarks but lacking generalization in practical applications.

---

We found no unified view among researchers in this field on how to solve this problem.

### Heatmap Regression Model

![arch_2.jpg](https://github.com/DocsaidLab/DocAligner/blob/main/docs/hmap_model_arch.jpg?raw=true)

This model retains the original feature extractor but modifies the Neck and Head parts.

1. **Feature Extraction**

   Besides using LCNet for mobile-friendly models, we also used a larger model to extract more features. We aim to create a model surpassing SoTA, and solely using LCNet is insufficient.

   In this model, we experimented with FastViT, MobileNetV2, and other "lightweight" convolutional neural networks, particularly suitable for efficient feature extraction in resource-constrained environments. We expect the Backbone to extract sufficient feature information from the input data, preparing for subsequent heatmap regression.

2. **BiFPN**

   To better integrate multi-scale features, we introduced BiFPN (Bidirectional Feature Pyramid Network), enhancing feature representation through bidirectional flow of contextual information. We expect BiFPN to produce a series of scale-rich and semantically strong feature maps, effectively capturing objects of different scales and positively impacting final prediction accuracy.

3. **Heatmap Regression**

   ![awing_loss.jpg](https://github.com/DocsaidLab/DocAligner/blob/main/docs/awing_loss.jpg?raw=true)

   To address the amplification error mentioned earlier, we need some "fuzziness" in the predicted results. This means the model should not precisely pinpoint the document's corner but indicate that "the corner is roughly in this area."

   For this, we adopted the common method in facial keypoint detection or human pose estimation: **heatmap regression**.

   Heatmap regression generates heatmap representations of objects, reflecting the likelihood of objects appearing at different locations. By analyzing these heatmaps, the model can accurately predict object positions and poses. In our scenario, the heatmaps are used to locate the document's corners.

4. **Adaptive Wing Loss**

   Loss is crucial in model training, responsible for calculating the difference between the model's predicted results and the actual labels.

   In this model, we use Adaptive Wing Loss, a loss function specifically for facial keypoint detection. This method innovates on the loss function for heatmap regression, particularly suitable for facial alignment problems. Its core idea is to let the loss function adjust its shape based on different types of true heatmap pixels, imposing more penalties on foreground pixels (pixels near facial feature points) and fewer penalties on background pixels.

   Here, we treat the document corner prediction problem as a facial keypoint detection problem and use a loss function designed for facial keypoint detection. We believe this method effectively addresses document corner detection issues and performs well in various scenarios.

   - **Reference:** [**Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression**](https://arxiv.org/abs/1904.07399)

   Besides the corner loss, we also used several auxiliary losses, including:

   - **Edge Loss:** Supervises object boundary information using AWing Loss.
   - **Classification Loss:** Used to predict whether a document exists in the image, using BCE Loss.

### Solving Amplification Error

The output of the heatmap regression model is a heatmap indicating where the document's corners are likely to be.

Next, we **cannot directly use this heatmap** because it is downscaled. The correct procedure should be:

1. Enlarge the heatmap to the original image size.
2. Use image post-processing to find the contours of the suggested corner areas in each heatmap.
3. Calculate the centroid of the contour, which is the corner of the document.

This way, the model can accurately locate the corners, solving the previously mentioned amplification error issue.

## Conclusion

You can find an apparent drawback in the heatmap model architecture:

- **It is not an end-to-end model architecture.**

This is also a question we constantly ponder when designing the model. We hope to design an end-to-end model, making it simpler for users and allowing the model to learn from each component. However, given the difficulties encountered with the point regression model, we had to use the heatmap regression model design.

In summary, although not perfect, it at least solves the amplification error issue.

In our heatmap model experiments, using a larger Backbone and a more complex Neck improves the model's accuracy.

In the deployment phase, you only need to consider the computing power limitations in the usage scenario and select an appropriate model.

## Citations

We thank all the predecessors whose work significantly helped our research.

If you find our work helpful, please cite our repository:

```bibtex
@misc{yuan2023docaligner,
  author = {Ze Yuan},
  title = {DocAligner},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/DocsaidLab/DocAligner},
  note = {GitHub repository}
}
```
