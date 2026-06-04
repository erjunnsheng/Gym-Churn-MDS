# main/ — 跑整個實驗會用到的腳本

從 0 重建整個專案資料 / 模型 / 圖檔的 4 個腳本。**從專案根目錄執行**：

```powershell
cd C:\Users\cihci\Desktop\MDS\gym_churn
.\.venv\Scripts\Activate.ps1

# 跑順序（依賴順序）
python main\make_notebook.py            # 1. 重建 analysis.ipynb（V1 完整 + V2 + Profiling SOP）
python main\experiment_v2.py             # 2. 訓練 V2 模型 → xgb_churn_model_v2.pkl
python main\experiment_clustering.py    # 3. 3 種分群方法 → clustering_models.pkl
python main\make_figures_v2.py          # 4. 產 figures_v2/ 給 app.py 用
```

## 4 個腳本各做什麼

| 腳本 | 輸入 | 輸出 |
|---|---|---|
| `make_notebook.py` | (cell 內容定義在此) | `analysis.ipynb` |
| `experiment_v2.py` | `data/gym_churn_us.csv` | `xgb_churn_model_v2.pkl` + V2 統計 |
| `experiment_clustering.py` | `data/gym_churn_us.csv` | `clustering_models.pkl`（K-means+GMM+Hier）|
| `make_figures_v2.py` | V2 模型 + 資料 | `figures_v2/` 10 張 PNG |

## 為什麼從根目錄跑？

腳本內所有路徑都是相對 cwd 的：
- 讀資料：`data/gym_churn_us.csv`
- 寫圖：`figures/...png`、`figures_v2/...png`
- 寫模型：`xgb_churn_model_v2.pkl`

從專案根目錄執行 = 所有相對路徑都對。

## 一鍵重建全部

```powershell
python main\make_notebook.py
python main\experiment_v2.py
python main\experiment_clustering.py
python main\make_figures_v2.py
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```
