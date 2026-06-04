"""Streamlit dashboard for the Gym Churn Risk Calculator.

Run:
    .\\.venv\\Scripts\\Activate.ps1
    streamlit run app.py
"""
import pickle

import numpy as np
import pandas as pd
import streamlit as st
# Imports needed for pickle.load to reconstruct fitted objects
from sklearn.cluster import KMeans, AgglomerativeClustering  # noqa: F401
from sklearn.mixture import GaussianMixture  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Gym Churn Risk Calculator",
    page_icon="🏋️",
    layout="wide",
)

# -----------------------------------------------------------------
# Constants (chocoZAP Taipei pricing)
# -----------------------------------------------------------------
MONTHLY_FEE = 1100
SIGNUP_FEE = 500

FEATURES = [
    "gender", "Near_Location", "Partner", "Promo_friends", "Phone",
    "Contract_period", "Group_visits", "Age",
    "Avg_additional_charges_total", "months_into_current",
    "Lifetime",
    "Avg_class_frequency_total", "Avg_class_frequency_current_month",
]

CLUSTER_STRATEGY = {
    "試水族": {
        "color": "#F44336",
        "tier": "高風險",
        "actions": [
            "📱 30 天打卡挑戰：8 次完課送一個月免費",
            "👥 同儕邀請：邀朋友各送 100 元現金折抵",
            "🕐 開放尖峰外時段不限預約長度",
            "📍 推送離家最近分店即時人流通知",
        ],
        "budget_pct": 0.20,
        "explain": "chocoZAP 模式的核心客群。個別 CLV 低，靠規模補。",
    },
    "熄火族": {
        "color": "#FF9800",
        "tier": "預警中",
        "actions": [
            "📨 教練個人化邀約 SMS / Line",
            "🎁 限時免費團體課邀請（低成本激活）",
            "📊 月報告：呈現過往進步、提醒目標",
            "❌ 避免群發續約信（會推他往退會方向想）",
        ],
        "budget_pct": 0.20,
        "explain": "時機比金額重要 — 早期 SMS 介入勝過後期挽留。",
    },
    "活躍短期": {
        "color": "#FFC107",
        "tier": "中等",
        "actions": [
            "💸 升級長約折扣（12 月一次付 85 折）",
            "🏃 跨分店通行強調（行動派福利）",
            "💪 個人教練體驗包（突破自學瓶頸）",
        ],
        "budget_pct": 0.25,
        "explain": "穩定但合約短。引導升級長約 = 降低未來流失。",
    },
    "核心 VIP": {
        "color": "#4CAF50",
        "tier": "低風險",
        "actions": [
            "🎁 年終回饋禮（毛巾、補給品）",
            "🤝 推薦獎勵：介紹新會員雙方各送月免費教練",
            "🚫 不要打擾 — 最好的留客是不要做奇怪的事",
        ],
        "budget_pct": 0.25,
        "explain": "CLV 最高。投資留客 ROI 最佳。",
    },
}


# -----------------------------------------------------------------
# Load model + fit clusterer once
# -----------------------------------------------------------------
@st.cache_resource
def load_resources():
    with open("xgb_churn_model_v2.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("clustering_models.pkl", "rb") as f:
        bundle = pickle.load(f)
    return xgb_model, bundle


xgb_model, bundle = load_resources()
scaler = bundle["scaler"]


def predict_cluster(method, scaled_row):
    """Return (cluster_id, name_map, soft_probs_or_None)."""
    if method == "K-means":
        m = bundle["kmeans"]
        cid = int(m["model"].predict(scaled_row)[0])
        return cid, m["names"], None
    elif method == "GMM":
        m = bundle["gmm"]
        cid = int(m["model"].predict(scaled_row)[0])
        probs = m["model"].predict_proba(scaled_row)[0]
        return cid, m["names"], probs
    else:  # Hierarchical (nearest centroid)
        m = bundle["hierarchical"]
        dists = np.linalg.norm(scaled_row[0] - m["centroids"], axis=1)
        cid = int(np.argmin(dists))
        return cid, m["names"], None


# -----------------------------------------------------------------
# Session state — for "Load my chocoZAP profile" button
# -----------------------------------------------------------------
DEFAULT_PROFILE = {
    "gender": 1, "Near_Location": 1, "Partner": 0, "Promo_friends": 0,
    "Phone": 1, "Contract_period": 6, "Group_visits": 1, "Age": 30,
    "Avg_additional_charges_total": 120.0,
    "months_into_current": 0.0, "Lifetime": 3,
    "Avg_class_frequency_total": 2.0,
    "Avg_class_frequency_current_month": 2.0,
}

CHOCOZAP_PROFILE = {
    "gender": 1,             # 預設男（你可改）
    "Near_Location": 1,      # 簽台北店
    "Partner": 0,            # 沒員工方案
    "Promo_friends": 0,      # 不是朋友推薦
    "Phone": 1,
    "Contract_period": 1,    # 月月制
    "Group_visits": 0,       # chocoZAP 沒傳統團體課
    "Age": 22,
    "Avg_additional_charges_total": 30.0,
    "months_into_current": 0.0,   # 4 月底剛簽，合約還沒過任何月
    "Lifetime": 1,                # 4 月底入會
    "Avg_class_frequency_total": 0.5,    # 4 週 2 次 = 0.5/週
    "Avg_class_frequency_current_month": 0.5,
}

if "profile" not in st.session_state:
    st.session_state.profile = DEFAULT_PROFILE.copy()


def load_chocozap():
    st.session_state.profile = CHOCOZAP_PROFILE.copy()


def reset_profile():
    st.session_state.profile = DEFAULT_PROFILE.copy()


# -----------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏋️ Gym Churn Calculator")
    st.markdown(
        """
        輸入會員特徵 → 即時預測：
        - 流失機率
        - 屬於哪一群客戶
        - 建議的留客策略 + 預算上限
        """
    )

    st.divider()
    st.markdown("### 🔬 分群方法")
    clustering_method = st.selectbox(
        "選擇分群演算法",
        options=["K-means", "GMM", "Hierarchical"],
        index=0,
        help=(
            "K-means: 經典中心式（baseline，churn 分布最分散）\n"
            "GMM: 機率式軟分群（會顯示屬於各群的機率）\n"
            "Hierarchical: 階層 Ward 連結（churn 較壓縮但結構穩定）"
        ),
    )
    method_summary = {
        "K-means": "群流失率 6/8/27/54%（最分散）",
        "GMM": "群流失率 7/14/40/40%（有兩個 40% 群）",
        "Hierarchical": "群流失率 15/27/30/41%（最壓縮）",
    }
    st.caption(f"📊 {method_summary[clustering_method]}")

    st.divider()
    st.markdown("### 🎯 載入真實案例")
    if st.button("📋 載入：我的 chocoZAP profile", use_container_width=True):
        load_chocozap()
        st.rerun()
    if st.button("🔄 重置為一般會員", use_container_width=True):
        reset_profile()
        st.rerun()

    st.divider()
    st.markdown("### 💵 計價基準")
    st.markdown(
        f"""
        **chocoZAP 台北店**
        - 入會費：NT$ {SIGNUP_FEE}
        - 月費：NT$ {MONTHLY_FEE}/月
        - 無綁約、隨時退
        """
    )

    st.divider()
    st.markdown("### 📚 模型（V2）")
    st.markdown(
        """
        - XGBoost (ROC AUC = 0.981)
        - K-means K=4 分群
        - 訓練資料：Yandex Practicum Model Fitness 4,000 筆
        - V2 特徵工程：以 `months_into_current`
          取代 `Month_to_end_contract`，
          多重共線性 0.97 → 0.44
        """
    )


# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("🏋️ Gym Churn Risk Calculator")
st.caption(
    "MDS 期末報告配套互動 demo (V2) — 輸入會員特徵，即時預測流失與分群，並依 chocoZAP 台北費率算 CLV 與建議行銷預算。"
)

p = st.session_state.profile

# -----------------------------------------------------------------
# Inputs (3 columns)
# -----------------------------------------------------------------
st.markdown("### 📥 輸入會員特徵")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 基本資料")
    age = st.slider("年齡", 18, 60, p["Age"])
    gender = st.radio("性別", [0, 1],
                      format_func=lambda x: "女" if x == 0 else "男",
                      index=p["gender"], horizontal=True)
    near = st.checkbox("住健身房附近", value=bool(p["Near_Location"]))
    phone = st.checkbox("有提供電話", value=bool(p["Phone"]))

with c2:
    st.markdown("#### 加入背景")
    partner = st.checkbox("公司員工方案 (Partner)", value=bool(p["Partner"]))
    promo = st.checkbox("朋友推薦加入 (Promo_friends)", value=bool(p["Promo_friends"]))
    group = st.checkbox("有上團體課 (Group_visits)", value=bool(p["Group_visits"]))
    contract = st.selectbox("合約長度（月）", [1, 6, 12],
                            index=[1, 6, 12].index(p["Contract_period"]))

with c3:
    st.markdown("#### 行為")
    lifetime = st.number_input("已入會月數 (Lifetime)", 0, 60, p["Lifetime"])
    months_into = st.number_input(
        "已過合約月數 (months_into_current)",
        min_value=0.0,
        max_value=float(contract),  # 不能超過 Contract_period
        value=min(float(p["months_into_current"]), float(contract)),
        step=0.5,
        help=f"= Contract_period − Month_to_end_contract。最大 = 合約長度 {contract} 月",
    )
    extra = st.number_input("月平均額外消費 (NT$)", 0.0, 1000.0,
                             float(p["Avg_additional_charges_total"]))
    freq_total = st.number_input("歷史平均上課頻率 (次/週)", 0.0, 7.0,
                                  float(p["Avg_class_frequency_total"]), step=0.1)
    freq_curr = st.number_input("本月上課頻率 (次/週)", 0.0, 7.0,
                                 float(p["Avg_class_frequency_current_month"]), step=0.1)
    # 反推 M2E 給 CLV 算（業務邏輯）
    month_end = float(contract) - months_into
    st.caption(f"📌 對應 Month_to_end_contract = {month_end:.1f} 月（剩餘合約）")

# -----------------------------------------------------------------
# Predict
# -----------------------------------------------------------------
row = pd.DataFrame([{
    "gender": int(gender), "Near_Location": int(near),
    "Partner": int(partner), "Promo_friends": int(promo),
    "Phone": int(phone), "Contract_period": int(contract),
    "Group_visits": int(group), "Age": int(age),
    "Avg_additional_charges_total": float(extra),
    "months_into_current": float(months_into),
    "Lifetime": int(lifetime),
    "Avg_class_frequency_total": float(freq_total),
    "Avg_class_frequency_current_month": float(freq_curr),
}])[FEATURES]

churn_proba = float(xgb_model.predict_proba(row)[0, 1])
scaled_row = scaler.transform(row)
cluster_id, CLUSTER_NAMES, soft_probs = predict_cluster(clustering_method, scaled_row)
cluster_name = CLUSTER_NAMES[cluster_id]
strategy = CLUSTER_STRATEGY[cluster_name]

# CLV (same formula as notebook Section 7)
historical_clv = SIGNUP_FEE + lifetime * MONTHLY_FEE + extra
expected_future_revenue = (1 - churn_proba) * max(month_end, 0) * MONTHLY_FEE
expected_clv = historical_clv + expected_future_revenue
budget_cap = expected_clv * strategy["budget_pct"]

# -----------------------------------------------------------------
# Results
# -----------------------------------------------------------------
st.divider()
st.markdown("### 📊 預測結果")

r1, r2, r3 = st.columns([1.1, 1.2, 1.7])

with r1:
    st.markdown("#### 流失風險")
    risk_color = "#F44336" if churn_proba > 0.5 else ("#FFC107" if churn_proba > 0.25 else "#4CAF50")
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; background:{risk_color}22;
                    border-radius:12px; border:3px solid {risk_color};">
            <div style="font-size:14px; color:#666;">流失機率</div>
            <div style="font-size:48px; font-weight:800; color:{risk_color};">
                {churn_proba:.0%}
            </div>
            <div style="font-size:12px; color:#666;">XGBoost 模型預測</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(churn_proba)

with r2:
    st.markdown("#### 客戶分群")
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; background:{strategy['color']}22;
                    border-radius:12px; border:3px solid {strategy['color']};">
            <div style="font-size:14px; color:#666;">客戶類型 ({clustering_method})</div>
            <div style="font-size:36px; font-weight:800; color:{strategy['color']};">
                {cluster_name}
            </div>
            <div style="font-size:13px; color:#444; margin-top:8px;">
                {strategy['tier']} · Cluster {cluster_id}
            </div>
            <div style="font-size:11px; color:#666; margin-top:6px;">
                {strategy['explain']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # GMM 獨有：軟分群機率
    if soft_probs is not None:
        st.markdown("##### 屬於各群的機率（GMM 軟分群）")
        sorted_probs = sorted(
            [(CLUSTER_NAMES[i], p) for i, p in enumerate(soft_probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        for name, p in sorted_probs:
            col_a, col_b = st.columns([2, 5])
            col_a.write(f"**{name}**")
            col_b.progress(float(p), text=f"{p:.1%}")

with r3:
    st.markdown("#### CLV + 建議預算")
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("歷史 CLV", f"${historical_clv:,.0f}")
    cv2.metric("預期未來營收", f"${expected_future_revenue:,.0f}")
    cv3.metric("總預期 CLV", f"${expected_clv:,.0f}")
    st.info(
        f"💰 **建議單客留客預算上限：NT$ {budget_cap:,.0f}** "
        f"（{strategy['budget_pct']:.0%} CLV）",
    )

# -----------------------------------------------------------------
# Recommended actions
# -----------------------------------------------------------------
st.divider()
st.markdown("### 💡 建議留客策略")

action_cols = st.columns(min(len(strategy["actions"]), 4))
for col, action in zip(action_cols, strategy["actions"]):
    with col:
        st.markdown(
            f"""
            <div style="padding:14px; background:#f5f5f5; border-radius:8px;
                        border-left:5px solid {strategy['color']}; min-height:90px;">
                {action}
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------
# chocoZAP case study (when this profile is loaded)
# -----------------------------------------------------------------
if (lifetime == CHOCOZAP_PROFILE["Lifetime"]
        and contract == CHOCOZAP_PROFILE["Contract_period"]
        and age == CHOCOZAP_PROFILE["Age"]
        and freq_curr == CHOCOZAP_PROFILE["Avg_class_frequency_current_month"]):
    st.divider()
    st.markdown("### 🎯 chocoZAP 真實案例對照")
    st.markdown(
        f"""
        **這是本報告作者的真實 chocoZAP 台北會員 profile**：

        - 2026/4 月底 1 元入會
        - 即被模型分到 **{cluster_name}**（cluster id={cluster_id}）
        - 預測流失機率 **{churn_proba:.0%}**
        - 預期 CLV 僅 **NT$ {expected_clv:,.0f}**

        **chocoZAP 的商業策略恰好是接受這個結果**：
        - 個別客戶 CLV 低 → 用 1 元入會降低 CAC，把 LTV/CAC 比拉回來
        - 預期流失高 → 用無綁約降低消費者抗拒
        - 日本一年內 880 家店、80 萬會員 → **規模補單客** 的教科書範例

        > 本作者親身的低留存行為，**驗證了模型對 Cluster 0 的判斷準確**。
        """
    )

# -----------------------------------------------------------------
# Feature importance reference
# -----------------------------------------------------------------
st.divider()
with st.expander("📈 模型 SHAP 特徵重要性（V2）"):
    try:
        st.image("figures_v2/v2_shap_bar.png", use_container_width=True)
        st.image("figures_v2/v2_shap_summary.png", use_container_width=True)
    except Exception as e:
        st.warning(f"找不到 SHAP 圖：{e}")

with st.expander("📊 客戶分群 PCA 視覺化（V2）"):
    try:
        st.image("figures_v2/v2_kmeans_pca.png", use_container_width=True)
        st.image("figures_v2/v2_churn_per_cluster.png", use_container_width=True)
    except Exception as e:
        st.warning(f"找不到分群圖：{e}")

with st.expander("🔬 三種分群方法對照（K-means vs GMM vs Hierarchical）"):
    try:
        st.image("figures_v2/v2_clustering_pca_compare.png", use_container_width=True)
        st.caption(
            "三個演算法在 PCA 2D 投影上的分群結果。"
            "**K-means** 把右側極端區整塊歸為試水族；"
            "**GMM** 用高斯橢圓切，產生兩個 40% churn 的相似群（重疊）；"
            "**Hierarchical** 大群偏中間，churn 分布最壓縮。"
        )
        st.image("figures_v2/v2_clustering_agreement.png", use_container_width=True)
        st.caption(
            "方法間 ARI 都 < 0.35 → 三個演算法看到的是**不同切法**，"
            "代表資料結構不存在單一「正確」分群。"
            "選擇 K-means 為主要方法因為它的群在 churn 維度上分得最開（最 actionable）。"
        )
        st.image("figures_v2/v2_dendrogram.png", use_container_width=True)
        st.caption("Hierarchical (Ward linkage) 的階層樹，紅線是切 K=4 的位置。")
    except Exception as e:
        st.warning(f"找不到分群方法對照圖：{e}")

with st.expander("💰 CLV 各群對比（V2）"):
    try:
        st.image("figures_v2/v2_clv_per_cluster.png", use_container_width=True)
        st.image("figures_v2/v2_clv_vs_churn_scatter.png", use_container_width=True)
    except Exception as e:
        st.warning(f"找不到 CLV 圖：{e}")

with st.expander("🔬 V1 vs V2 ROC 對照"):
    try:
        st.image("figures_v2/v1_vs_v2_roc.png", use_container_width=True)
        st.caption(
            "V2 把 Month_to_end_contract 替換成 months_into_current，"
            "AUC 從 0.9796 微升到 0.9809，但相關係數從 0.97 降到 0.44 — "
            "更乾淨的特徵空間。"
        )
    except Exception as e:
        st.warning(f"找不到對照圖：{e}")

# Footer
st.divider()
st.caption(
    "MDS 期末報告 · Gym Churn Analysis (V2) · "
    "資料：Yandex Practicum Model Fitness · "
    "費率：chocoZAP 台北店 · "
    "模型：XGBoost (V2, ROC AUC=0.981) + K-means"
)
