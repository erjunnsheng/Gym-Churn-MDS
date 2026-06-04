# tools/ — 工具腳本

不是跑實驗主流程必須的，但是分析、診斷、輸出產物時很有用的小工具。**從專案根目錄執行**：

```powershell
cd C:\Users\cihci\Desktop\MDS\gym_churn
.\.venv\Scripts\Activate.ps1
python tools\<腳本名稱>.py
```

## 6 個腳本

### 📊 簡報 / 文件
| 腳本 | 做什麼 |
|---|---|
| `build_pptx.py` | 用 python-pptx 產出 `MDS_期末報告.pptx`（20 slides） |

### 🔍 分析 / 診斷
| 腳本 | 做什麼 |
|---|---|
| `trace_cluster_deep.py` | K-means 4 群的 6 個角度深度分析（z-score、原型成員、兩兩 cluster 差異、PCA loadings） |
| `trace_kmeans_k.py` | K=2..8 的 Elbow / Silhouette / Davies-Bouldin 多指標對照 |
| `trace_shap_local.py` | 全域 SHAP vs 個別 SHAP 對照 + perturbation sensitivity |
| `trace_my_profile.py` | 本作者 chocoZAP profile 通過 V2 pipeline 的完整 trace |

### ✅ 資料驗證
| 腳本 | 做什麼 |
|---|---|
| `verify_time_features.py` | 檢查 `Contract_period` / `Month_to_end_contract` / `Lifetime` 的邏輯一致性 |

## 報告章節對應

| 報告章節 | 用哪個腳本的輸出 |
|---|---|
| 第七章 方法論：分群分析 | `trace_cluster_deep.py` |
| 第七章 方法論：K 值選擇 | `trace_kmeans_k.py` |
| 第七章 SHAP 進階：全域 vs 個別 | `trace_shap_local.py` |
| 第十章 反思：資料品質 | `verify_time_features.py` |
| 個案研究：本作者 profile | `trace_my_profile.py` |
| 簡報 pptx | `build_pptx.py` |

## 共通約定

- 全部從專案根目錄執行
- 全部使用 venv：`.\.venv\Scripts\Activate.ps1`
- 寫圖到 `figures/` 或 `figures_v2/`（相對 cwd）
- 讀資料 / 模型 `data/`、`*.pkl`（相對 cwd）
