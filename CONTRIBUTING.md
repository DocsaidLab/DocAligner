# 為 Docsaid 做出貢獻

首先，非常感謝您考慮為 Docsaid 做出貢獻。正是像您這樣的人，使得 Docsaid 成為如此出色的工具。

## 接下來我應該做什麼？

如果您發現了一個錯誤或有功能需求，請確定檢查我們的問題區，看看社群中是否已經有人建立了相同的議題。如果沒有，請 **創建一個**！

## 分叉 (Fork) 和創建一個分支

如果您覺得自己能夠解決這個問題，那就分叉存儲庫並創建一個描述性名字的分支吧。

一個好的分支名稱應該如下（其中問題 #325 是您正在處理的議題）：

```bash
git checkout -b feature/325-add-japanese-localization
```

## 運行測試套件

確保您能夠運行測試套件。如果您在運行它時遇到困難，請創建一個問題，我們會幫助您開始。

## 實施您的修復或功能

此時，您已經準備好進行變更了！如果需要幫助，請隨時向我們求助；每個人一開始都是初學者。

## 創建 Pull Request

此時，您應該切換回主分支，並確保它與 Docsaid 的主分支保持最新：

```bash
git remote add upstream git@github.com:original/docsaidlab.git
git checkout main
git pull upstream main
```

然後從您的本地主分支更新您的功能分支，並推送它！

```bash
git checkout feature/325-add-japanese-localization
git rebase main
git push --set-upstream origin feature/325-add-japanese-localization
```

前往 Docsaid 存儲庫，您應該可以看到最近推送的分支。

選擇您的分支並創建一個新的 Pull Request。一旦您創建了 Pull Request，維護者會審查您的變更。

## 保持您的 Pull Request 更新

如果維護者要求您 “rebase” 您的 PR，這意味著有很多代碼已經更改，您需要更新您的分支以便更容易合併。

## 合併 PR（僅限維護者）

只有在以下所有條件都滿足的情況下，維護者才能將 PR 合併到主分支：

- 它通過了 CI（持續集成）。
- 它已經得到至少兩個維護者的批准。如果是維護者開啟的 PR，只需要一個額外的批准。
- 它沒有要求的更改。
- 它與當前的主分支保持最新。

任何維護者在所有這些條件都滿足的情況下都有權合併 PR。

## 感謝您的貢獻！

非常感謝您的時間和努力。謝謝您為 Docsaid 做出的貢獻！
