# DocAligned

DocAligned 專案是一個專注於文本識別和定位的計算機視覺系統。其核心功能是準確地偵測和識別圖像中文本的四個角點，這對於各種圖像處理和文本分析應用來說可以，或可以降低後續任務的錯誤和運算量。

主要功能和特點包括：

- **文本角點識別**：DocAligned 能夠識別圖像中文本區域的精確位置，特別是文本區塊的四個角點。這對於後續的文本提取和分析非常重要，特別是在需要精確切割和識別文本的場景中。
- **多樣化的文本處理**：系統對於不同類型的文本具有良好的識別能力，包括不同字體、大小和顏色的文本。它可以有效處理從標準文本到複雜的手寫筆記等各種形式的文本。
- **適應不同環境**：DocAligned 在處理含有複雜背景或不規則排列的文本時表現出色，能夠有效分辨和定位文本與其他視覺元素之間的關係。
- **技術實現**：這個系統利用深度學習模型來學習從圖像中識別文本的特徵。通過訓練，模型能夠適應不同的文本樣式和布局，提高識別的準確性。
- **靈活部署**：使用 Docker 容器化技術，DocAligned 可以在不同的系統和平台上進行快速部署，確保在不同環境中都能保持一致的運行效果。

## 資料集概覽

- **MIDV-500/MIDV-2019：**
   - 倉庫：[MIDV-500 GitHub 頁面](https://github.com/fcakyon/midv500)
   - 一個全面的文件影像分析資料集。

- **MIT室內場景：**
   - 詳情：[MIT 室內場景](https://web.mit.edu/torralba/www/indoor.html)
   - 包含多樣化室內環境影像。

- **CORD v0:**
   - 倉庫：[CORD GitHub 頁面](https://github.com/clovaai/cord)
   - 專注於OCR任務的收據影像資料集。

## 安裝和設定

### 先決條件：

- 確保安裝了Python。
- 檢查資料集頁面上的任何特定係統需求。

### 步驟：

1. **安裝 MIDV-500 套件：**

    ```bash
    pip install midv500
    ```

2. **下載資料集：**

    - **MIDV-500/MIDV-2019：**
      安裝後執行 `download_midv.py`。

      ```bash
      python download_midv.py
      ```

    - **MIT室內場景 & CORD v0:**
      訪問各自的鏈接，並按照其下載說明操作。

3. **建構資料集：**

    把 MIDV 和 CORD 資料集放在同一個地方，並在 `build_dataset.py` 中設定 `ROOT` 變數為儲存資料集的目錄。確認完成後，執行以下命令：

    ```bash
    python build_dataset.py
    ```

## `docpool` 資料夾說明

- 此資料夾包含從網路收集的文字影像，將在後續使用動態合成影像技術作為訓練資料集。
- 後續腳本將利用這些影像進行資料擴充和處理。

## 構建訓練環境

首先，請您確保已經從 `DocsaidKit` 內建置了基礎映像 `docsaid_training_base_image`。如果您還沒有這樣做，請先參考 `DocsaidKit` 的說明文件。接著，請使用以下指令來建置 DocAligned 工作的 Docker 映像：

```bash
cd DocAligned
bash docker/build.bash
```

## 執行訓練（Based on Docker）

這部分的說明如何利用您已經構建的 Docker 映像來執行文檔對齊訓練。

請退到 `DocAligned` 的上層目錄，並執行以下指令來啟動訓練：

```bash
bash DocAligned/docker/train.bash LC150_BIFPN64_D3_PointReg_r256 # 這裡替換成您的配置文件名稱
```

通過這些步驟，您可以在 Docker 容器內安全地執行文檔對齊訓練任務，同時利用 Docker 的隔離環境來確保一致性和可重現性。這種方法使得項目的部署和擴展變得更加方便和靈活。

