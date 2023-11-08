# Dataset

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
