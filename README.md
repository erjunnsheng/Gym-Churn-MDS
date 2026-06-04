# 健身房會員流失預測 + STP 客戶分群 + 留客策略

MDS 期末報告配套技術專案。對 *Model Fitness* 健身房 4000 筆會員資料做：
EDA → 流失預測（Logistic / RF / XGBoost）→ SHAP 解釋 → K-means 分群 → **CLV 計算（chocoZAP 台北費率）** → **4 個人物誌（含本作者真實 chocoZAP Cluster 0 案例）** → **Feature Engineering V1 vs V2 對照實驗（共線性 0.97 → 0.44）** → **互動式 Streamlit dashboard (V2 模型)**。

## 環境

- Windows 11
- Python 3.11.9
- 套件鎖在 `requirements.txt`（pip freeze 全量）
- venv 已建好在 `.venv/`

## 啟動方式

### A. 跑分析 notebook

```powershell
cd C:\Users\cihci\Desktop\MDS\gym_churn
.\.venv\Scripts\Activate.ps1
jupyter notebook analysis.ipynb
```

開瀏覽器後按 `Cell → Run All`，約 40 秒跑完。

### B. 啟動 Streamlit 互動 demo

```powershell
cd C:\Users\cihci\Desktop\MDS\gym_churn
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

瀏覽器自動打開 http://localhost:8501，左側選單可以 **載入本作者的 chocoZAP profile** 一鍵試。

> 注意：`app.py` 載入的是 **V2 模型** (`xgb_churn_model_v2.pkl`)，輸入特徵為 `months_into_current` 而非 `Month_to_end_contract`。V1 模型 (`xgb_churn_model.pkl`) 仍保留供對照。詳見 notebook §7.5。

### C. 純命令列重跑 notebook（不開 UI）

```powershell
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```

## 專案結構

```
gym_churn/
├── data/
│   ├── gym_churn_us.csv            # 原始資料（4000 筆 × 14 欄）
│   ├── gym_churn_with_clusters.csv # 加上 K-means cluster 欄
│   ├── cluster_profile.csv         # V1 4 群側寫摘要
│   └── cluster_profile_v2.csv      # V2 4 群側寫摘要
├── figures/                        # 17 張 PNG，直接放進報告
│   ├── 01_churn_distribution.png
│   ├── 02_numeric_features_by_churn.png
│   ├── 03_binary_features_churn_rate.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_roc_comparison.png
│   ├── 06_confusion_matrices.png
│   ├── 07_shap_summary.png
│   ├── 08_shap_bar.png
│   ├── 09_shap_dependence.png
│   ├── 10_elbow.png
│   ├── 11_kmeans_pca.png
│   ├── 12_churn_per_cluster.png
│   ├── 13_clv_per_cluster.png
│   ├── 14_clv_vs_churn_scatter.png
│   ├── 15_v1_v2_correlation.png    ← §7.5 V1 vs V2 共線性熱圖
│   ├── 16_v1_v2_roc.png            ← §7.5 6 條 ROC 對照
│   └── 17_v1_v2_shap_bar.png       ← §7.5 SHAP 並列比較
├── figures_v2/                     ← app.py 顯示用的 V2 圖
│   ├── v1_vs_v2_roc.png
│   ├── v2_shap_bar.png / summary
│   ├── v2_kmeans_pca.png
│   ├── v2_churn_per_cluster.png
│   ├── v2_clv_per_cluster.png
│   └── v2_clv_vs_churn_scatter.png
├── analysis.ipynb                  # 主分析筆記本（77 cells、含 §7.5 V1 vs V2 實驗）
├── make_notebook.py                # 用 Python 重建 ipynb
├── experiment_v2.py                # V1 vs V2 對照實驗腳本
├── make_figures_v2.py              # 產生 figures_v2/ 的腳本
├── app.py                          # Streamlit 互動 dashboard（V2 模型）
├── personas.md                     # 4 個客戶人物誌（Cluster 0 = 本作者）
├── xgb_churn_model.pkl             # V1 XGBoost
├── xgb_churn_model_v2.pkl          # V2 XGBoost（app.py 載入）
├── requirements.txt                # pip freeze 全量鎖版本
└── README.md
```

## Notebook 章節對照（77 cells）

| 章節 | 內容 |
|---|---|
| 1. 載入資料 | shape / info / describe / 缺值與重複檢查 |
| 2. EDA | 流失率分布、數值特徵 by churn、二元特徵流失率、相關性熱圖 |
| 3. 預處理 | train/test split + StandardScaler |
| 4. 模型訓練 | Logistic / RF / XGBoost + ROC + Confusion Matrix |
| 5. SHAP | 樹模型可解釋性：summary + bar + dependence |
| 6. K-means | Elbow 選 K + 4 群側寫 + PCA 視覺化 + 各群流失率 |
| 7. **CLV** ⭐ | chocoZAP 1100/月 費率 → 各群歷史 + 預期 CLV → 行銷預算上限 + chocoZAP 案例對照 |
| **7.5 Feature Engineering V1 vs V2** ⭐ | 用 `months_into_current` 取代 `Month_to_end_contract` 的對照實驗：共線性 0.97 → 0.44，三模型 AUC 持平或微升，群結構穩定 |
| 8. 商業建議 | 對應報告章節 + 未來工作 |

## ⭐ 給 MDS 報告的章節對應

| 報告章節 | 從哪裡拿 |
|---|---|
| **第二章 現況** | personas.md Cluster 0（本作者 chocoZAP 真實故事） |
| **第三章 資料產品** | 整個 pipeline + `app.py` Streamlit 互動 demo |
| **第六章 4P · Product** | XGBoost + SHAP + dashboard |
| **第六章 4P · Price** | Notebook Section 7.1-7.3 CLV 結果 + 各群定價建議 |
| **第六章 4P · Promotion** | personas.md 各群行銷預算 + 行動清單 |
| **第七章 技術** | 三模型 ROC 比較 + SHAP top features + 分群方法論 + CLV 模型 + §7.5 V1 vs V2 對照實驗 |
| **第八章 STP** | K-means 4 群 = Segmentation；CLV 排序 = Target；personas.md 4 個人物 = Positioning 對象 |
| **個案研究章節** ⭐ | Notebook Section 7.4 chocoZAP 商業模式對照 + 本人 Cluster 0 命中經驗 |
| **附錄 / 實驗紀錄** ⭐ | Notebook §7.5 完整 V1 vs V2 對照（共線性、ROC、SHAP、群結構） |
| **第十章 反思** | 多重共線性（V2 已部分解決 0.97 → 0.44）、缺乏滿意度資料、未做 uplift modeling、可加 NPS |

## chocoZAP 為什麼是好案例

| 我們的模型告訴你 | chocoZAP 的策略 |
|---|---|
| Cluster 0 試水族流失率 56% | **接受**：不靠單客賺、靠量取勝 |
| Cluster 0 預期 CLV 低 | **降低 CAC**：1 元入會 + 500 元入會費 |
| 留客 ROI 不划算 | **無綁約**降低消費者抗拒 |
| 個別客戶價值低 | **規模補單客**：日本 1 年 880 家店 / 80 萬會員 |

→ 「**STP 鎖定 Cluster 0、靠商業模式取勝**」的教科書級案例。
→ 本作者親身就是被 chocoZAP 命中又流失的真實樣本，加強報告的可信度與 PM 思維。

## 可改的閾值與參數

| 想改什麼 | 改 notebook 的哪一段 |
|---|---|
| 月費 / 入會費 | Section 7 開頭 `MONTHLY_FEE` / `SIGNUP_FEE` |
| K-means 的 K 值 | Section 6.2 `K = 4` |
| XGBoost 超參數 | Section 4 的 `xgb.XGBClassifier(...)` |
| SHAP top N | Section 5.3 `[:3]` 改 `[:5]` |
| 不同 train/test 切分 | Section 3 `RANDOM_STATE` |
| 留客預算比例 | Section 7.3 `0.20` / `0.30` |

## 已知限制（寫進反思）

- 沒有滿意度 / NPS 資料 → 缺「為什麼會留下」的維度
- `Avg_class_frequency_total` 與 `current_month` 仍有多重共線性（V2 未處理這對）
- ✅ Contract_period ↔ M2E 共線性 V2 已處理（0.97 → 0.44，見 §7.5）
- 資料為 Yandex Practicum 出版的教學集（虛構但統計合理）
- 未做 uplift modeling（無實驗組資料）
- 未針對不平衡做 SMOTE / class_weight 校正（27% churn 還可接受）
- CLV 用單一月費假設、未做不同方案組合
- Cluster 0 persona 訪談 n=1（只有本作者），未擴大樣本
- 137 筆樣本時間欄位 rounding 不一致（資料粒度限制）

## 故障排解

| 症狀 | 解法 |
|---|---|
| `jupyter` 找不到 | `Activate.ps1` 沒跑或 venv 沒裝 |
| `streamlit` 找不到 | 同上；或 `pip install streamlit` |
| Streamlit port 被占 | 改用 `streamlit run app.py --server.port 8502` |
| 中文字變方塊 | `Microsoft JhengHei` 字型沒裝（Win11 預設有） |
| SHAP 圖在 Streamlit 顯示不出來 | 重跑 notebook（圖檔可能還沒生成） |
| `MissingIDFieldWarning` | 無害，不影響執行 |

## 如果要改流程重建 notebook

`make_notebook.py` 是 notebook 的「原始碼」：

```powershell
.\.venv\Scripts\python.exe make_notebook.py
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```

## 資料來源

- [Kaggle - Model Fitness Customer Churn](https://www.kaggle.com/datasets/ellanihill/model-fitness-customer-churn)
- 鏡像下載 URL：[github.com/dmytrovoytko/ml-churn-prediction](https://github.com/dmytrovoytko/ml-churn-prediction)
- chocoZAP 台北店費率：[chocozap.tw/lp/main-01](https://chocozap.tw/lp/main-01/)
- chocoZAP 商業背景：[經理人雜誌報導](https://www.managertoday.com.tw/articles/view/67946)
