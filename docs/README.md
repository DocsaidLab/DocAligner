**[English](../README.md)** | **[中文](./README.md)**

# DocAligned

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligned/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligned?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## 介紹

本專案是一個專注於證件（文件）等各類文本定位之視覺系統。我們對於該系統的期待主要為提供圖像中文本的四個角點之預測。這項功能在面對金融科技、銀行及共享經濟服務時的應用中至關重要，對於各種圖像處理和文本分析應用來說，可以降低後續任務的錯誤和運算量。

## 目錄
- [介紹](#介紹)
- [目錄](#目錄)
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
      cd DocAligned/data
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
)

# 只有 SmartDoc 2015 資料集有第三個回傳值，用來作為驗證集與 benchmark 所使用。
img, poly, _ = ds[0]
D.imwrite(D.draw_polygon(img, poly, thickness=5), 'smartdoc_test_img.jpg')
```

<div align="center">
    <img src="./smartdoc_test_img.jpg" width="300">
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
    <img src="./midv500_test_img.jpg" width="300">
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
    <img src="./midv2019_test_img.jpg" width="300">
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
    <img src="./midv2020_test_img.jpg" width="300">
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
    <img src="./cordv0_test_img.jpg" width="300">
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
    <img src="./sync_test_img.jpg" width="300">
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

## 構建訓練環境

首先，請您確保已經從 `DocsaidKit` 內建置了基礎映像 `docsaid_training_base_image`。如果您還沒有這樣做，請先參考 `DocsaidKit` 的說明文件。接著，請使用以下指令來建置 DocAligned 工作的 Docker 映像：

```bash
cd DocAligned
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

首先，看到 `train.bash` 檔案內容：

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

針對上述檔案的說明如下：

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
   - `-v $PWD/DocAligned:/code/DocAligned` 等：這些是掛載參數，將主機的目錄映射到容器內部的目錄，以便於訓練數據和腳本的訪問。
   - `--cfg_name $1`：這是傳遞給 `trainer.py` 的參數，指定了配置文件的名稱。

3. **數據集路徑**
   - 特別注意 `/data/Dataset` 是用於存放訓練數據的路徑，您會需要調整 `-v /data/Dataset:/data/Dataset` 這段指令，把 `/data/Dataset` 替換成您的資料集目錄。

最後，請退到 `DocAligned` 的上層目錄，並執行以下指令來啟動訓練：

```bash
bash DocAligned/docker/train.bash LC150_BIFPN64_D3_PointReg_r256 # 這裡替換成您的配置文件名稱
```

- 補充：配置文件說明可以參考 [DocAligned/model/README.md](./model/README.md)。

通過這些步驟，您可以在 Docker 容器內安全地執行文檔對齊訓練任務，同時利用 Docker 的隔離環境來確保一致性和可重現性。這種方法使得項目的部署和擴展變得更加方便和靈活。

