[English](./README_en.md) | **[中文](./README.md)**

# DocAligner

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligner/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligner?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## 介紹

本項目是一個專注於定位圖像中文件的視覺系統，我們主要目標是提供文件四個角點的預測。該功能主要應用在金融科技、銀行業和共享經濟等領域中，它能夠降低各種圖像處理和文字分析任務的錯誤率和計算需求。

<div align="center">
    <img src="./docs/title.jpg" width="800">
</div>


## 目錄

- [介紹](#介紹)
- [目錄](#目錄)
- [安裝](#安裝)
- [使用說明](#使用說明)
    - [導入必要的依賴項](#導入必要的依賴項)
    - [ModelType](#modeltype)
    - [Backend](#backend)
    - [創建 DocAligner 實例](#創建-docaligner-實例)
    - [讀取和處理圖像](#讀取和處理圖像)
    - [結果處理](#結果處理)
        - [繪製文檔多邊形](#繪製文檔多邊形)
        - [取得繪製後的 numpy 圖像](#取得繪製後的-numpy-圖像)
        - [提取攤平後的文檔圖像](#提取攤平後的文檔圖像)
        - [將文檔資訊轉為 JSON](#將文檔資訊轉為-json)
        - [完整範例](#完整範例)
- [Benchmark](#benchmark)
- [訓練模型](#訓練模型)
    - [資料集介紹](#資料集介紹)
    - [資料集預處理](#資料集預處理)
    - [資料集實作](#資料集實作)
        - [1. SmartDoc 2015 資料集](#1-smartdoc-2015-資料集)
        - [2. MIDV-500 資料集](#2-midv-500-資料集)
        - [3. MIDV-2019 資料集](#3-midv-2019-資料集)
        - [4. MIDV-2020 資料集](#4-midv-2020-資料集)
        - [5. CORD v0 資料集](#5-cord-v0-資料集)
        - [6. 合成資料集](#6-合成資料集)
        - [7. 影像增強](#7-影像增強)
- [構建訓練環境](#構建訓練環境)
- [執行訓練（Based on Docker）](#執行訓練based-on-docker)
- [轉換模型為 ONNX 格式（Based on Docker）](#轉換模型為-onnx-格式based-on-docker)

---

## 安裝

目前我們沒有提供 Pypi 上的安裝包，若要使用本專案，您可以直接從 Github 上 clone 本專案，然後安裝相依套件，安裝前請確認您已經安裝了 [DocsaidKit](https://github.com/DocsaidLab/DocsaidKit)。

若已經安裝 DocsaidKit，請按照以下步驟進行：

1. Clone 專案：

   ```bash
   git clone https://github.com/DocsaidLab/DocAligner.git
   ```

2. 進入專案目錄：

   ```bash
   cd DocAligner
   ```

3. 建立打包文件：

   ```bash
   python setup.py bdist_wheel
   ```

4. 安裝打包文件：

   ```bash
   pip install dist/docaligner-*-py3-none-any.whl
   ```

遵循這些步驟，您應該能夠順利完成 DocAligner 的安裝。

安裝完成後即可以使用本專案。

---

## 使用說明

我們提供了一個簡單的模型推論介面，其中包含了前後處理的邏輯。

首先，您需要導入所需的相關依賴並創建 DocAligner 類別。

### 導入必要的依賴項

```python
import docsaidkit as D
from docsaidkit import Backend
from docaligner import DocAligner, ModelType
```

### ModelType

`ModelType` 是一個枚舉類型，用於指定 DocAligner 使用的模型類型。它包含以下選項：

- `heatmap`：使用熱圖模型進行文檔對齊。
- `point`：使用點檢測模型進行文檔對齊。

未來可能會有更多的模型類型，我們會在此處更新。

### Backend

`Backend` 是一個枚舉類型，用於指定 DocAligner 的運算後端。它包含以下選項：

- `cpu`：使用 CPU 進行運算。
- `cuda`：使用 GPU 進行運算（需要適當的硬體支援）。

ONNXRuntime 支援了非常多的後端，包括 CPU、CUDA、OpenCL、DirectX、TensorRT 等等，若您有其他需求，可以參考 [**ONNXRuntime Execution Providers**](https://onnxruntime.ai/docs/execution-providers/index.html)，並自行修改成對應的後端。

### 創建 DocAligner 實例

```python
model = DocAligner(
    gpu_id=0,  # GPU 編號，如果不使用 GPU 請設為 -1
    backend=Backend.cpu,  # 選擇運算後端，可以是 Backend.cpu 或 Backend.cuda
    model_type=ModelType.point  # 選擇模型類型，可以是 ModelType.heatmap 或 ModelType.point
)
```

注意事項：

- 使用 cuda 運算除了需要適當的硬體支援外，還需要安裝相應的 CUDA 驅動程式和 CUDA 工具包。如果您的系統中沒有安裝 CUDA，或安裝的版本不正確，則無法使用 CUDA 運算後端。

- 關於 onnxruntime 安裝依賴相關的問題，請參考 [ONNXRuntime Release Notes](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

### 讀取和處理圖像

```python
# 讀取圖像
img = D.imread('path/to/your/image.jpg')

# 您也可以使用我們提供的測試圖像
# img = D.imread('docs/run_test_card.jpg')

# 使用模型進行推論
result = model(img) # result 是一個 Document 類型
```

### 結果處理

您得到的推論結果是經過我們包裝的 `Document` 類型，它包含了文檔的多邊形、OCR 文字資訊等等。

`Document` 類別提供了多種功能，以協助處理和分析文件圖像。主要功能包括：

1. **文件多邊形處理**：能夠辨識和操作文件的邊界。
2. **OCR 文字辨識**：支援從圖像中辨識文字。
3. **圖像變形**：能夠根據文件的邊界轉換圖像。

- 屬性
    - `image`：存儲文件的圖像。
    - `doc_polygon`：文件的多邊形邊界。
    - `ocr_texts`：OCR 辨識出的文字列表。
    - `ocr_polygons`：與 `ocr_texts` 相對應的多邊形邊界。

- 方法
    - `gen_doc_flat_img()`：將文件圖像根據其多邊形邊界變形。
    - `gen_doc_info_image()`：生成一個標記了文件邊界和方向的圖像。
    - `gen_ocr_info_image()`：生成一個顯示 OCR 文字和其邊界的圖像。
    - `draw_doc()`：將標記了文件邊界的圖像保存到指定路徑。
    - `draw_ocr()`：將標記了 OCR 文字和邊界的圖像保存到指定路徑。

在這個模組中，我們不會用到 OCR 相關的功能，因此我們只會使用 `image` 和 `doc_polygon` 屬性。獲取到推論結果後，您可以進行多種後處理操作。

#### 繪製文檔多邊形

```python
# 繪製並保存帶有文檔多邊形的圖像
result.draw_doc('path/to/save/folder', 'output_image.jpg')
```

或不指定保存路徑，則會在當前目錄下保存，並自動給定一個時序編號。

```python
result.draw_doc()
```

#### 取得繪製後的 numpy 圖像

使用 `draw_doc` 功能預設會保存 JPG 格式的圖像，如果您有其他需求，可以使用 `gen_doc_info_image` 方法，之後再自行處理。

```python
img = result.gen_doc_info_image()
```

#### 提取攤平後的文檔圖像

如果您知道文檔的原始大小，即可以使用 `gen_doc_flat_img` 方法，將文檔圖像根據其多邊形邊界轉換為矩形圖像。

```python
H, W = 1080, 1920
flat_img = result.gen_doc_flat_img(image_size=(H, W))
```

如果是一個未知的影像類別，也可以不用給定 `image_size` 參數，此時將會根據文檔多邊形的邊界自動計算出最小的矩形圖像，並將最小矩形的長寬設為 `H` 和 `W`。

```python
flat_img = result.gen_doc_flat_img()
```

#### 將文檔資訊轉為 JSON

如果您需要將文檔資訊保存到 JSON 檔案中，可以使用 `be_jsonable` 方法。

轉換時，可以考慮將影像剔除，以節省空間，預設使用 `exclude_image=True`。

```python
doc_json = result.be_jsonable()
D.dump_json(doc_json)
```

#### 完整範例

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

不只是你，我們也想知道我們的模型效果如何，所以我們使用了 [SmartDoc 2015](https://github.com/jchazalon/smartdoc15-ch1-dataset) 資料集作為我們的測試資料集。

### 評估協議

我們使用 **Jaccard Index** 作為衡量標準，這個指數總結了不同方法在正確分割頁面輪廓方面的能力，並對那些在某些畫面中未能檢測到文件對象的方法進行了懲罰。

評估過程首先是利用每個畫面中文件的大小和坐標，將提交方法 S 和基準真實 G 的四邊形坐標進行透視變換，以獲得校正後的四邊形 S0 和 G0。這樣的變換使得所有的評估量度在文件參考系內是可比的。對於每個畫面 f，計算 Jaccard 指數 (JI)，這是一種衡量校正四邊形重疊程度的指標，計算公式如下：

$$ JI(f) = \frac{\text{area}(G0 \cap S0)}{\text{area}(G0 \cup S0)} $$

其中 $` \text{area}(G0 \cap S0) `$ 定義為檢測到的四邊形和基準真實四邊形的交集多邊形，$` \text{area}(G0 \cup S0) `$ 則為它們的聯集多邊形。每種方法的總體分數將是測試數據集中所有畫面分數的平均值。

### 評估結果

現階段的模型表現還沒有到 SoTA 的分數，但是已經可以滿足大部分的應用場景。

以 `PointRec-256` 的模型來說，目前的開發規模為 6 MB，運算量約為 1.2 GFLPOs，因此可以在各種設備上運行，包括手機、嵌入式設備等等。

以 `PointRec-512` 的模型來說，模型尺寸一樣，但輸入影像解析度加倍，因此運算量加四倍，約為 5.0 GFLPOs。

我們認為訓練方式和資料集的構成方式，都是影響模型效果的重要因素，因此我們會持續更新模型，並且提供更多的資料集，以提升模型的效果。

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

## 訓練模型

我們不提供模型微調的功能，但是您可以使用我們的訓練模組，自行產出模型。

大致上來說，需要遵循幾個步驟：

1. 準備資料集
2. 構建訓練環境
3. 執行訓練
4. 評估模型
5. 轉換模型為 ONNX 格式
6. 評估是否需要量化
7. 將 ONNX 模型更新到專案中打包

以下我們將逐步說明。

---

在開始說明前，我們也體諒到您可能只有經費，而沒有足夠的時間來思考如何對您的現場環境進行對應的客製化調整。因此您也可以直接聯絡我們進行諮詢，我們可以根據您經費的多寡和施工難度，幫您調度工程師來進行客製化的開發。

聯絡方式：docsaidlab@gmail.com

---

## 資料集介紹

- **SmartDoc 2015**
    - [**SmartDoc 2015**](https://github.com/jchazalon/smartdoc15-ch1-dataset)
    - Smartdoc 2015 - Challenge 1 資料集最初是為 Smartdoc 2015 競賽創建的，重點是評估使用智慧型手機的文件影像擷取方法。 挑戰 1 特別在於偵測和分割從智慧型手機預覽串流中擷取的視訊畫面中的文件區域。

- **MIDV-500/MIDV-2019**
   - [**MIDV**](https://github.com/fcakyon/midv500)
   - MIDV-500 由 50 個不同身分證明文件類型的500 個影片片段組成，包括 17 個身分證、14 個護照、13 個駕照和 6 個不同國家的其他身分證明文件，並具有真實性，可以對各種文件分析問題進行廣泛的研究。
   - MIDV-2019 資料集包含扭曲和低光影像。

- **MIDV-2020:**
   - [**MIDV2020**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
   - MIDV-2020 包含 10 種文件類型，其中包括 1000 個帶註釋的影片剪輯、1000 個掃描影像和 1000 個獨特模擬身分文件的 1000 張照片，每個文件都具有唯一的文字欄位值和唯一的人工生成的面孔。

- **Indoor Scenes**
   - [**Indoor**](https://web.mit.edu/torralba/www/indoor.html)
   - 該資料集包含 67 個室內類別，總共 15,620 張圖像。圖像數量因類別而異，但每個類別至少有 100 張圖像。所有圖片均為 jpg 格式。

- **CORD v0**
   - [**CORD**](https://github.com/clovaai/cord)
   - 該資料集由數千張印尼收據組成，其中包含用於 OCR 的圖像文字註釋，以及用於解析的多層語義標籤。所提出的資料集可用於解決各種 OCR 和解析任務。

- **Docpool**
   - [**Docpool**](./data/docpool/)
   - 我們自行從網路收集各類文本影像，用在動態合成影像技術作為訓練資料集。


## 資料集預處理

1. **安裝 MIDV-500 套件：**

    ```bash
    pip install midv500
    ```

2. **下載資料集：**

    - **MIDV-500/MIDV-2019：**
      安裝後執行 `download_midv.py`。

      ```bash
      cd DocAligner/data
      python download_midv.py
      ```

    - **MIDV-2020：**
      訪問各自的鏈接，並按照其下載說明操作。

    - **SmartDoc 2015：**
      訪問各自的鏈接，並按照其下載說明操作。

    - **室內場景 & CORD v0:**
      訪問各自的鏈接，並按照其下載說明操作。

3. **建構資料集：**

    把 MIDV 和 CORD 資料集放在同一個地方，並在 `build_dataset.py` 中設定 `ROOT` 變數為儲存資料集的目錄。確認完成後，執行以下命令：

    ```bash
    python build_dataset.py
    ```

   完成後，會產生多個 `.json` 檔案。這些檔案包含了資料集的所有資訊，包括圖像路徑、標籤、圖像大小等等。

## 資料集實作

我們針對上述的幾個資料集，進行對應於 pytorch 訓練的資料集實作，請參考 [dataset.py](./model/dataset.py)。

以下我們實際展示如何讀取資料集：

### 1. SmartDoc 2015 資料集

```python
import docsaidkit as D
from model.dataset import SmartDocDataset

ds = SmartDocDataset(
    root="/data/Dataset" # Replace with your dataset directory
    mode="val", # "train" or "val"
    train_ratio=0.2 # Using 20% of the data for training and 80% for validation.

# 只有 SmartDoc 2015 資料集有第三個回傳值，用來作為驗證集與 benchmark 所使用。
img, poly, doc_type = ds[0]

# 如果設定 `mode="train"`，則只會回傳前兩個值。
# img, poly = ds[0]

D.imwrite(D.draw_polygon(img, poly, thickness=5), 'smartdoc_test_img.jpg')
```

<div align="center">
    <img src="./docs/smartdoc_test_img.jpg" width="500">
</div>

### 2. MIDV-500 資料集

```python
import docsaidkit as D
from model.dataset import MIDV500Dataset

ds = MIDV500Dataset(
    root="/data/Dataset" # 請替換成您的資料集目錄
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'midv500_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv500_test_img.jpg" width="300">
</div>

### 3. MIDV-2019 資料集

```python
import docsaidkit as D
from model.dataset import MIDV2019Dataset

ds = MIDV2019Dataset(
    root="/data/Dataset" # 請替換成您的資料集目錄
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'midv2019_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv2019_test_img.jpg" width="300">
</div>

### 4. MIDV-2020 資料集

```python
import docsaidkit as D
from model.dataset import MIDV2020Dataset

ds = MIDV2020Dataset(
    root="/data/Dataset" # 請替換成您的資料集目錄
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=3), 'midv2020_test_img.jpg')
```

<div align="center">
    <img src="./docs/midv2020_test_img.jpg" width="300">
</div>


### 5. CORD v0 資料集

```python
import docsaidkit as D
from model.dataset import CordDataset

ds = CordDataset(
    root="/data/Dataset" # 請替換成您的資料集目錄
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'cordv0_test_img.jpg')
```

<div align="center">
    <img src="./docs/cordv0_test_img.jpg" width="300">
</div>

### 6. 合成資料集

考慮到資料集的不足，我們使用動態合成影像技術。

簡單來說，我們先收集了一份 Docpool 資料集，其中包含了從網路上找到的各類證件和文件的影像。接著，我們找來了 Indoor 資料集作為背景，然後將 Docpool 內的資料，合成到背景上。

此外，MIDV-500/MIDV-2019/CORD 資料集中，也都有對應的 Polygon 資料，秉持著不浪費的精神，我們也會將 Docpool 內的圖片合成到這些資料集上，以增加資料集的多樣性。

總之，拿來用就對了，實作細節什麼的，您不感興趣就直接放到一邊就好。

```python
import docsaidkit as D
from model.dataset import SyncDataset

ds = SyncDataset(
    root="/data/Dataset" # 請替換成您的資料集目錄
)

img, poly = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=2), 'sync_test_img.jpg')
```

<div align="center">
    <img src="./docs/sync_test_img.jpg" width="300">
</div>


### 7. 影像增強

儘管我們已經收集了一些的資料，但是這些資料的多樣性仍然不足。為了增加資料的多樣性，我們使用了影像增強技術，這些技術可以模擬圖像在拍攝時的各種情況，例如遮擋、移動、旋轉、模糊、噪聲、顏色變化等等。

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
   - 這個增強技術會隨機在圖像中產生一個矩形區域，並將該區域內的像素值設置為 0。可以模擬圖像中的遮擋，例如圖像中的文字被其他物體遮擋的情況。

- **GaussianBlur**
    - 這個增強技術會對圖像進行高斯模糊。可以模擬圖像在拍攝時的高斯模糊，次外，合成影像會有比較銳利的邊緣特徵，這個增強技術可以模糊邊緣特徵。讓他們看起來更像真實的影像。

- **Others**
    - 這些增強技術可以模擬圖像在拍攝時的各種情況，例如移動、旋轉、模糊、噪聲、顏色變化等等。

---

## 構建訓練環境

首先，請您確保已經從 `DocsaidKit` 內建置了基礎映像 `docsaid_training_base_image`。如果您還沒有這樣做，請先參考 `DocsaidKit` 的說明文件。接著，請使用以下指令來建置 DocAligner 工作的 Docker 映像：

```bash
# Build base image from docsaidkit at first
git clone https://github.com/DocsaidLab/DocsaidKit.git
cd DocsaidKit
bash docker/build.bash

# Then build DocAligner image
git clone https://github.com/DocsaidLab/DocAligner.git
cd DocAligner
bash docker/build.bash
```

這是我們預設採用的 [Dockerfile](./docker/Dockerfile)，專門為執行文件對齊訓練設計，您可以根據自己的需求進行修改，以下是該文件的說明：

1. **基礎鏡像**
    - `FROM docsaid_training_base_image:latest`
    - 這行指定了容器的基礎鏡像，即 `docsaid_training_base_image` 的最新版本。基礎映像像是建立您的 Docker 容器的起點，它包含了預先配置好的作業系統和一些基本的工具，您可以在 `DocsaidKit` 的專案中找到它。

2. **工作目錄設定**
    - `WORKDIR /code`
    - 這裡設定了容器內的工作目錄為 `/code`。 工作目錄是 Docker 容器中的一個目錄，您的應用程式和所有的命令都會在這個目錄下運作。

3. **環境變數**
    - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
    - 這行定義了一個環境變數 `ENTRYPOINT_SCRIPT`，其值設定為 `/entrypoint.sh`。 環境變數用於儲存常用配置，可以在容器的任何地方存取。

4. **安裝 gosu**
    - 透過 `RUN` 指令安裝了 `gosu`。 `gosu` 是一個輕量級的工具，允許使用者以特定的使用者身分執行命令，類似於 `sudo`，但更適合 Docker 容器。
    - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` 這行指令首先更新了套件列表，然後安裝`gosu`，最後清理了不再需要 的檔案以減小鏡像大小。

5. **建立入口點腳本**
    - 透過一系列 `RUN` 指令建立了入口點腳本 `/entrypoint.sh`。
    - 此腳本首先檢查環境變數 `USER_ID` 和 `GROUP_ID` 是否被設定。 如果設定了，腳本會建立一個新的使用者和使用者群組，並以該使用者身分執行命令。
    - 這對於處理容器內外檔案權限問題非常有用，特別是當容器需要存取宿主機上的檔案時。

6. **賦予權限**
    - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` 這行指令使入口點腳本成為可執行檔。

7. **設定容器的入口點和預設指令**
    - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` 和 `CMD ["bash"]`
    - 這些命令指定了容器啟動時執行的預設命令。 當容器啟動時，它將執行 `/entrypoint.sh` 腳本。

## 執行訓練（Based on Docker）

這部分的說明如何利用您已經構建的 Docker 映像來執行文檔對齊訓練。

首先，請您看到 `train.bash` 檔案內容：

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

針對上述檔案的說明如下，如果您想要動手修改的話，可以參考一下：

1. **創建訓練腳本**
   - `cat > trainer.py <<EOF ... EOF`
   - 這段命令創建了一個 Python 腳本 `trainer.py`。這個腳本導入了必要的模塊和函數，並在腳本的主部分中調用 `main_docalign_train` 函數。使用 Google's Python Fire 庫來解析命令行參數，使得命令行界面的生成更加容易。

2. **運行 Docker 容器**
   - `docker run ... doc_align_train python trainer.py --cfg_name $1`
   - 這行命令啟動了一個 Docker 容器，並在其中運行 `trainer.py` 腳本。
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`：這些參數將當前用戶的用戶 ID 和組 ID 傳遞給容器，以便在容器內創建具有相應權限的用戶。
   - `--gpus all`：指定容器可以使用所有 GPU。
   - `--shm-size=64g`：設置共享內存的大小，這在大規模數據處理時很有用。
   - `--ipc=host --net=host`：這些設置允許容器使用主機的 IPC 命名空間和網絡堆棧，有助於性能提升。
   - `--cpuset-cpus="0-31"`：指定容器使用哪些 CPU 核心。
   - `-v $PWD/DocAligner:/code/DocAligner` 等：這些是掛載參數，將主機的目錄映射到容器內部的目錄，以便於訓練數據和腳本的訪問。
   - `--cfg_name $1`：這是傳遞給 `trainer.py` 的參數，指定了配置文件的名稱。

3. **數據集路徑**
   - 特別注意 `/data/Dataset` 是用於存放訓練數據的路徑，您會需要調整 `-v /data/Dataset:/data/Dataset` 這段指令，把 `/data/Dataset` 替換成您的資料集目錄。

最後，請退到 `DocAligner` 的上層目錄，並執行以下指令來啟動訓練：

```bash
bash DocAligner/docker/train.bash lcnet100_point_reg_bifpn # 這裡替換成您的配置文件名稱
```

- 補充：配置文件說明可以參考 [DocAligner/model/README.md](./model/README.md)。

通過這些步驟，您可以在 Docker 容器內安全地執行文檔對齊訓練任務，同時利用 Docker 的隔離環境來確保一致性和可重現性。這種方法使得項目的部署和擴展變得更加方便和靈活。

## 轉換模型為 ONNX 格式（Based on Docker）

這部分的說明如何利用您的模型轉換為 ONNX 格式。

首先，請您看到 `to_onnx.bash` 檔案內容：

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

從這個檔案開始看起，但不需要修改它，您需要去修改對應的檔案：`model/to_onnx.py`

在訓練過程中，您可能會使用許多分支來監督模型的訓練，但是在推論階段，您可能只需要其中的一個分支。因此，我們需要將模型轉換為 ONNX 格式，並且只保留推論階段所需要的分支。

例如：

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

在上面這個範例中，我們只取出推論用的分支，並且將其封裝為一個新的模型 `WarpLC100FPN`。接著，在 yaml config 上進行相對應的參數設定：

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

說明模型的輸入尺寸，輸入名稱，輸出名稱，以及 ONNX 的版本號。

轉換的部分我們已經幫您寫好了，完成上面的修改後，確認 `model/to_onnx.py` 檔案有指向您的模型，並且退到 `DocAligner` 的上層目錄，並執行以下指令來啟動轉換：

```bash
bash DocAligner/docker/to_onnx.bash lcnet100_point_reg_bifpn # 這裡替換成您的配置文件名稱
```

這時候，您會在 `DocAligner/model` 目錄下看到一個新的 ONNX 模型，把這個模型搬到 `docaligner/xxx` 對應的推論模型目錄下，就可以進行推論了。
