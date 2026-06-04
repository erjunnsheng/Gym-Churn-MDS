"""驗證 Contract_period / Month_to_end_contract / Lifetime 三個欄位的關係。"""
import sys
import io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_csv("data/gym_churn_us.csv")

# ----- 1. 各欄位的值域 -----
print("=" * 70)
print("1. 各欄位的值域")
print("=" * 70)
for col in ["Contract_period", "Month_to_end_contract", "Lifetime"]:
    vals = df[col]
    uniq = sorted(vals.unique())
    print(f"\n{col}")
    print(f"  範圍: [{vals.min()}, {vals.max()}]")
    print(f"  平均: {vals.mean():.2f}, 中位: {vals.median():.1f}")
    print(f"  唯一值數: {len(uniq)}")
    if len(uniq) <= 15:
        print(f"  全部值: {uniq}")
    else:
        print(f"  前 15: {uniq[:15]}")

# ----- 2. Month_to_end_contract <= Contract_period 嗎？ -----
print("\n" + "=" * 70)
print("2. Month_to_end_contract <= Contract_period 嗎？")
print("=" * 70)
violations = df[df["Month_to_end_contract"] > df["Contract_period"]]
print(f"違反這個關係的列數: {len(violations)} / {len(df)}")
if len(violations) > 0:
    print("\n違反者前 5 筆：")
    print(violations[["Contract_period", "Month_to_end_contract", "Lifetime"]].head())
else:
    print("→ 完全成立：Month_to_end_contract 一律 <= Contract_period")

# ----- 3. Lifetime vs Contract_period -----
print("\n" + "=" * 70)
print("3. Lifetime > Contract_period 的人有多少？（暗示他們續約過）")
print("=" * 70)
renewed = df[df["Lifetime"] > df["Contract_period"]]
print(f"Lifetime > Contract_period 的人: {len(renewed)} / {len(df)} = {len(renewed)/len(df):.1%}")

# 對 Contract_period = 1 月的會員看看 Lifetime 分布
print("\n各 Contract_period 對應的 Lifetime 分布:")
for cp in sorted(df["Contract_period"].unique()):
    sub = df[df["Contract_period"] == cp]
    print(f"  Contract_period = {cp:>4d}  → Lifetime min={sub['Lifetime'].min():>3.0f}  "
          f"median={sub['Lifetime'].median():>4.1f}  max={sub['Lifetime'].max():>4.0f}  "
          f"mean={sub['Lifetime'].mean():.2f}  n={len(sub)}")

# ----- 4. 樣本檢視 -----
print("\n" + "=" * 70)
print("4. 看幾筆具體資料來感受")
print("=" * 70)
samples = df[["Contract_period", "Month_to_end_contract", "Lifetime",
              "Avg_class_frequency_total", "Churn"]].sample(
    n=15, random_state=42).sort_values(
    ["Contract_period", "Lifetime"])
print(samples.to_string())

# ----- 5. 邏輯推斷 -----
print("\n" + "=" * 70)
print("5. 邏輯推斷：Contract_period - Month_to_end_contract = 「這次合約已過月數」")
print("=" * 70)
df["months_into_current"] = df["Contract_period"] - df["Month_to_end_contract"]
print("\nmonths_into_current 的分布：")
print(df["months_into_current"].describe())
print(f"\n有負值（不合理）的數量: {(df['months_into_current'] < 0).sum()}")
print(f"等於 0（剛簽合約）的數量: {(df['months_into_current'] == 0).sum()}")

# ----- 6. Lifetime 是否 >= months_into_current 嗎？ -----
print("\n" + "=" * 70)
print("6. 驗證 Lifetime >= months_into_current（一定要的）")
print("=" * 70)
viol = df[df["Lifetime"] < df["months_into_current"]]
print(f"違反者: {len(viol)} 筆")
if len(viol) > 0:
    print("這代表「在這次合約裡度過的時間」比「總會員時間」還長 → 邏輯不通")
    print(viol[["Contract_period", "Month_to_end_contract", "Lifetime",
                "months_into_current"]].head(10))
else:
    print("→ 完全成立：所有人的 Lifetime 都 >= 已過合約月數")

# ----- 7. 相關係數 -----
print("\n" + "=" * 70)
print("7. 三個時間相關欄位的相關係數")
print("=" * 70)
corr = df[["Contract_period", "Month_to_end_contract", "Lifetime"]].corr()
print(corr.round(3))
