"""Generate MDS_期末報告.pptx from project files.

Strict rules:
  - 所有數字來自 README / personas.md / cluster_profile_v2.csv / V2 實驗
  - 找不到就標 〔TODO：待補〕，不杜撰
  - 直接嵌入 figures/ 內 PNG，不重畫
  - 不重跑 notebook、不改其他檔案
"""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

ROOT = Path(".")
OUT = ROOT / "MDS_期末報告.pptx"

# ---- design tokens ----
FONT = "Microsoft JhengHei"
PRIMARY = RGBColor(0x1F, 0x3A, 0x5F)   # 深藍主色
ACCENT  = RGBColor(0xE5, 0x39, 0x35)   # 紅色強調
GREY    = RGBColor(0x55, 0x55, 0x55)
LIGHT_BG = RGBColor(0xF2, 0xF2, 0xF2)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
TODO_BG = RGBColor(0xFF, 0xF3, 0xCD)
TODO_BD = RGBColor(0xE0, 0xB7, 0x00)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ---- helpers ----
def set_font(run, name=FONT):
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", name)

def add_text(slide, text, left, top, width, height, *,
             size=18, bold=False, color=PRIMARY, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    set_font(run)
    return tb

def add_bullets(slide, items, left, top, width, height, *,
                size=20, color=PRIMARY, line_space=1.15):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_space
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = "•  " + txt
        run.font.size = Pt(size)
        run.font.color.rgb = color
        set_font(run)
    return tb

def add_image(slide, path, left, top, width=None, height=None):
    p = ROOT / path
    if not p.exists():
        return add_placeholder(slide, f"〔TODO：找不到 {path}〕",
                                left, top,
                                width or Inches(4), height or Inches(2.5))
    return slide.shapes.add_picture(str(p), left, top, width=width, height=height)

def add_placeholder(slide, text, left, top, width, height, *, color=TODO_BG, border=TODO_BD):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.color.rgb = border
    sh.line.width = Pt(1.5)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1); tf.margin_bottom = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(16)
    run.font.color.rgb = GREY
    run.font.italic = True
    set_font(run)
    return sh

def add_page_number(slide, n):
    add_text(slide, f"{n}", SLIDE_W - Inches(0.65), SLIDE_H - Inches(0.4),
             Inches(0.5), Inches(0.3), size=11, color=GREY, align=PP_ALIGN.RIGHT)

def add_footer_brand(slide):
    add_text(slide, "MDS 期末報告 · Gym Churn × chocoZAP",
             Inches(0.5), SLIDE_H - Inches(0.4), Inches(8), Inches(0.3),
             size=10, color=GREY, align=PP_ALIGN.LEFT)

def add_title_bar(slide, title, subtitle=None):
    # accent stripe on the left
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                     0, 0, Inches(0.25), SLIDE_H)
    stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT
    stripe.line.fill.background()

    add_text(slide, title, Inches(0.55), Inches(0.3),
             Inches(12.5), Inches(0.75), size=32, bold=True, color=PRIMARY)
    # underline
    underline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.55), Inches(1.1),
                                        Inches(2), Inches(0.05))
    underline.fill.solid(); underline.fill.fore_color.rgb = ACCENT
    underline.line.fill.background()
    if subtitle:
        add_text(slide, subtitle, Inches(0.55), Inches(1.15),
                 Inches(12.5), Inches(0.5), size=16, color=GREY, italic=True)

def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def add_divider(prs, part_label, part_title):
    s = blank(prs)
    # full background
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid(); bg.fill.fore_color.rgb = PRIMARY
    bg.line.fill.background()
    # accent bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(3),
                              Inches(0.15), Inches(1.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    add_text(s, part_label, Inches(1.4), Inches(3), Inches(10), Inches(0.6),
             size=22, color=ACCENT, bold=True)
    add_text(s, part_title, Inches(1.4), Inches(3.6), Inches(11), Inches(1.2),
             size=44, color=WHITE, bold=True)
    return s


# =====================================================================
# Build slides
# =====================================================================
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# ---------- Slide 1: Cover ----------
s = blank(prs)
# top + bottom accent stripes
top_stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.5))
top_stripe.fill.solid(); top_stripe.fill.fore_color.rgb = ACCENT
top_stripe.line.fill.background()
bot_stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, SLIDE_H - Inches(0.5), SLIDE_W, Inches(0.5))
bot_stripe.fill.solid(); bot_stripe.fill.fore_color.rgb = PRIMARY
bot_stripe.line.fill.background()

add_text(s, "MDS 期末報告", Inches(0.8), Inches(1.4),
         Inches(11.5), Inches(0.7), size=24, color=GREY)
add_text(s, "健身房會員流失預測與留客策略",
         Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.4),
         size=46, bold=True, color=PRIMARY)
add_text(s, "—— 以 chocoZAP 為例",
         Inches(0.8), Inches(3.55), Inches(11.7), Inches(0.8),
         size=30, color=ACCENT, bold=True)
# subtitle bullet
add_text(s,
         "Model Fitness (4000 筆會員) · XGBoost · K-means K=4 · CLV · Streamlit demo",
         Inches(0.8), Inches(4.6), Inches(11.7), Inches(0.5),
         size=16, color=GREY, italic=True)
# author + date placeholders
add_placeholder(s, "〔TODO：姓名〕", Inches(0.8), Inches(6),
                Inches(5.5), Inches(0.6))
add_placeholder(s, "〔TODO：日期〕", Inches(7), Inches(6),
                Inches(5.5), Inches(0.6))

add_notes(s,
    "本次報告主題：用 4000 筆健身房會員資料做流失預測與分群，"
    "並以 chocoZAP 為個案，呼應 MDS『行銷 + 統計 + 程式』。"
    "我本人就是其中的真實 Cluster 0 樣本。"
)

# ---------- Divider 1 ----------
add_divider(prs, "第二部分", "chocoZAP 與我的故事")

# ---------- Slide 2: chocoZAP intro ----------
s = blank(prs)
add_title_bar(s, "chocoZAP 是什麼？",
              "日本超低價便利健身房 — 1 元入會的商業實驗")
add_bullets(s, [
    "日本 RIZAP 集團子品牌，2022/7 推出",
    "目標客群官方原話:「沒有運動習慣的人」、「開始接觸運動的人」",
    "台北月費 NT$ 1,100、入會費 NT$ 500、限時 1 元入會促銷",
    "無綁約、隔月起停收、APP 自助、無人店模式",
    "店內含 KTV、按摩椅、高爾夫等非典型健身設施",
    "日本一年內 880 家店、80 萬會員 — 規模擴張驚人",
], Inches(0.55), Inches(1.85), Inches(7.5), Inches(5), size=20)
add_placeholder(s,
    "〔插入 chocoZAP 圖片〕\n（避免版權，請自行加入官網或店面照）",
    Inches(8.4), Inches(2), Inches(4.5), Inches(3.3))
add_page_number(s, 2)
add_footer_brand(s)
add_notes(s,
    "chocoZAP 用『便利商店』模式進入健身產業，極低門檻吸客。"
    "這套商業模式跟傳統健身房很不一樣，是後面個案研究的主角。"
    "下一頁講我本人為什麼會被吸引、又為什麼會流失。"
)

# ---------- Slide 3: 我的故事 ----------
s = blank(prs)
add_title_bar(s, "我的故事：被 1 元吸引、又被營運勸退",
              "本作者 = 模型裡的 Cluster 0「試水族」真實樣本")
# 三個吸引點 callout
add_text(s, "🎯 加入動機", Inches(0.55), Inches(1.85),
         Inches(6), Inches(0.5), size=22, bold=True, color=ACCENT)
add_bullets(s, [
    "🔓 不綁約 — 不喜歡可以隨時退",
    "🎤 店內 KTV — 傳統健身房沒有的招牌設施",
    "🤖 全自助 — 不用換衣服、APP 掃條碼進場",
    "限時 1 元入會 → 心理門檻幾乎為零",
], Inches(0.55), Inches(2.4), Inches(6), Inches(3), size=18)

add_text(s, "🚪 為什麼流失（第一手訪談）",
         Inches(7), Inches(1.85), Inches(6), Inches(0.5),
         size=22, bold=True, color=ACCENT)
add_bullets(s, [
    "基礎器材不足，重訓族品項不齊",
    "器材品質不如預期（NT 1,100/月期待落差）",
    "招牌設施（KTV、按摩椅）APP 永遠約不到",
    "→ 行銷端承諾 vs 營運端兌現的落差",
], Inches(7), Inches(2.4), Inches(6), Inches(3), size=18)

# 結尾大字
add_text(s,
    "❓ 像我這樣的會員為什麼會流失？健身房能預測並留住嗎？",
    Inches(0.55), Inches(6.3), Inches(12.2), Inches(0.7),
    size=20, bold=True, color=PRIMARY, align=PP_ALIGN.CENTER)
add_page_number(s, 3)
add_footer_brand(s)
add_notes(s,
    "我 2026/4 月底用 1 元簽 chocoZAP 台北店，5 月只去過 2 次。"
    "模型預測我流失機率 88.4%（V1）/ 83.1%（V2），事實上我確實放棄使用。"
    "這個故事直接引到接下來的整套資料科學分析。"
)

# ---------- Divider 2 ----------
add_divider(prs, "第三部分", "專案介紹")

# ---------- Slide 4: 專案概覽 (NEW - Exec Summary / Intro / About Data / Key Findings) ----------
s = blank(prs)
add_title_bar(s, "專案概覽",
              "Executive Summary · Introduction · About the Data · Key Findings")

LEFT = Inches(0.55)
LW = Inches(7.3)

add_text(s, "📋 Executive Summary", LEFT, Inches(1.75),
         LW, Inches(0.35), size=16, bold=True, color=ACCENT)
add_text(s,
    "對 Model Fitness 健身房 4,000 筆會員資料，建立 "
    "流失預測 (XGBoost) + K-means 分群 + CLV 留客預算 的完整 pipeline，"
    "並以 chocoZAP 為個案研究。",
    LEFT, Inches(2.15), LW, Inches(0.9),
    size=13, color=PRIMARY)

add_text(s, "🎯 Introduction", LEFT, Inches(3.05),
         LW, Inches(0.35), size=16, bold=True, color=ACCENT)
add_text(s,
    "健身房的核心挑戰：用最低成本留住最該留的會員。"
    "本研究建立「預測 → 為什麼 → 分群 → 定價」的可執行流程，"
    "對應 STP / 4P 行銷框架。",
    LEFT, Inches(3.45), LW, Inches(0.9),
    size=13, color=PRIMARY)

add_text(s, "📊 About the Data", LEFT, Inches(4.35),
         LW, Inches(0.35), size=16, bold=True, color=ACCENT)
add_text(s,
    "Yandex Practicum「Model Fitness」資料集  •  4,000 列 × 14 欄  "
    "•  6 個二元特徵 + 7 個數值特徵 + 1 個 Churn 標籤  "
    "•  無缺值、無重複  •  整體流失率 ≈ 27%",
    LEFT, Inches(4.75), LW, Inches(0.9),
    size=13, color=PRIMARY)

add_text(s, "💡 Key Findings", LEFT, Inches(5.65),
         LW, Inches(0.35), size=16, bold=True, color=ACCENT)
add_bullets(s, [
    "XGBoost ROC AUC = 0.9809（V2）",
    "K-means 4 群流失率差距明顯：6% / 8% / 27% / 54%",
    "V2 特徵工程：共線性 0.97 → 0.44，預測力不減",
    "本作者 chocoZAP profile 預測流失 83% → 實際確實流失",
], LEFT, Inches(6.05), LW, Inches(1.4), size=12, line_space=1.05)

add_placeholder(s,
    "〔TODO：插入資料格式截圖〕\n\n"
    "建議內容：\n"
    "df.head() / CSV schema / 欄位列表",
    Inches(8.1), Inches(1.75), Inches(4.85), Inches(5.7))

add_page_number(s, 4)
add_footer_brand(s)
add_notes(s,
    "專案標準化開場（Executive Summary / Introduction / About the Data / Key Findings），"
    "讓老師 30 秒內掌握整個專案的價值與資料。"
    "右側資料格式截圖請手動補一張 df.head() 或 CSV schema 截圖。"
)


# ---------- Slide 5: 專案目標與資料 (RESTORED ORIGINAL CONTENT) ----------
s = blank(prs)
add_title_bar(s, "專案目標與資料",
              "用資料科學回答四個關鍵商業問題")
add_bullets(s, [
    "❶ 誰會流失？  → XGBoost 預測（個體）",
    "❷ 為什麼會流失？  → SHAP 可解釋性",
    "❸ 客戶可分為哪幾種？  → K-means 分群 (STP)",
    "❹ 不同人該花多少預算留？  → CLV (4P · Price)",
], Inches(0.55), Inches(1.85), Inches(7.5), Inches(3.5), size=22)

# 資料 box
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                          Inches(8.4), Inches(1.85),
                          Inches(4.5), Inches(4.5))
box.fill.solid(); box.fill.fore_color.rgb = LIGHT_BG
box.line.color.rgb = PRIMARY
box.line.width = Pt(1)
add_text(s, "📊 資料集", Inches(8.6), Inches(2),
         Inches(4.2), Inches(0.5), size=18, bold=True, color=PRIMARY)
add_bullets(s, [
    "Yandex Practicum",
    "「Model Fitness」",
    "4,000 筆會員 × 14 欄",
    "13 個特徵 + Churn 標籤",
    "整體流失率 ≈ 27%",
    "無缺值、無重複",
], Inches(8.6), Inches(2.6), Inches(4.3), Inches(3.7), size=15)
add_page_number(s, 6)
add_footer_brand(s)
add_notes(s,
    "這份資料是 Yandex 出版的教學集（虛構但統計合理），對 MVP 是合理選擇。"
    "後面反思章節會提到『未來工作』可以接真實 chocoZAP / 健身工廠資料。"
)

# ---------- Slide 5: EDA ----------
s = blank(prs)
add_title_bar(s, "探索性資料分析（EDA）",
              "流失分布 + 數值/二元特徵 + 相關性熱圖")
add_image(s, "figures/01_churn_distribution.png",
          Inches(0.5), Inches(1.85), width=Inches(3.0))
add_image(s, "figures/02_numeric_features_by_churn.png",
          Inches(3.7), Inches(1.85), width=Inches(5.5))
add_image(s, "figures/03_binary_features_churn_rate.png",
          Inches(0.5), Inches(4.4), width=Inches(4.6))
add_image(s, "figures/04_correlation_heatmap.png",
          Inches(5.3), Inches(4.4), width=Inches(3.0))
add_bullets(s, [
    "整體流失 ≈ 27%",
    "年紀小、合約短、頻率低 → 流失高",
    "住附近、有員工方案、上團體課 → 顯著留客",
    "freq_total ↔ current_month 高度共線",
], Inches(8.5), Inches(4.4), Inches(4.5), Inches(2.7), size=15)
add_page_number(s, 5)
add_footer_brand(s)
add_notes(s,
    "EDA 階段先看了流失率分布、各特徵 by churn、二元特徵的影響、以及全特徵相關矩陣。"
    "其中『年資、頻率、合約』三組是後面 SHAP 排名最前的主因。"
)

# ---------- Slide 6: 三模型比較 ----------
s = blank(prs)
add_title_bar(s, "流失預測：三模型比較",
              "Logistic / Random Forest / XGBoost（V2 特徵）")
add_image(s, "figures/05_roc_comparison.png",
          Inches(0.55), Inches(1.85), width=Inches(6.5))
add_image(s, "figures/06_confusion_matrices.png",
          Inches(0.55), Inches(5.2), width=Inches(8))

# 指標表（V2 數字）
add_text(s, "📈 ROC AUC (V2)", Inches(8), Inches(1.85),
         Inches(5), Inches(0.5), size=20, bold=True, color=PRIMARY)
add_bullets(s, [
    "XGBoost      0.9809  ⭐ 最佳",
    "Logistic     0.9774",
    "Random Forest 0.9692",
    "三模型 AUC 皆 ≥ 0.96",
    "PR-AUC: 〔TODO：notebook 未計算〕",
], Inches(8), Inches(2.4), Inches(5), Inches(3), size=16)
add_page_number(s, 7)
add_footer_brand(s)
add_notes(s,
    "XGBoost 微勝 Logistic 跟 RF，但差距很小。"
    "選 XGBoost 為正式上線模型；Logistic 因係數可解釋，留作內部報告用。"
    "PR-AUC 是補充指標，目前 notebook 沒算 — 是 TODO。"
)

# ---------- Slide 7: SHAP ----------
s = blank(prs)
add_title_bar(s, "模型可解釋性（SHAP）",
              "流失主因排名 + 個別 feature 的 dependence")
add_image(s, "figures/08_shap_bar.png",
          Inches(0.55), Inches(1.85), width=Inches(6))
add_image(s, "figures/07_shap_summary.png",
          Inches(6.85), Inches(1.85), width=Inches(6))
add_image(s, "figures/09_shap_dependence.png",
          Inches(0.55), Inches(5.0), width=Inches(12.3))
add_text(s,
    "Top 5 流失主因 (V2)：current_month 頻率 · Lifetime · total 頻率 · "
    "Contract_period · Age",
    Inches(0.55), Inches(7.0), Inches(12.3), Inches(0.4),
    size=14, color=GREY, italic=True, align=PP_ALIGN.CENTER)
add_page_number(s, 8)
add_footer_brand(s)
add_notes(s,
    "SHAP 顯示前三大流失主因都跟『時間維度』有關：本月頻率掉、入會時間短、歷史頻率低。"
    "這直接呼應行銷部能介入的時機 — 越早期介入越有效。"
)

# ---------- Slide 8: V1 vs V2 ----------
s = blank(prs)
add_title_bar(s, "特徵工程：V1 vs V2",
              "把 Month_to_end_contract 換成 months_into_current")
add_image(s, "figures/15_v1_v2_correlation.png",
          Inches(0.55), Inches(1.85), width=Inches(6.3))
add_image(s, "figures/16_v1_v2_roc.png",
          Inches(7), Inches(1.85), width=Inches(6))

add_text(s, "💡 結果", Inches(0.55), Inches(5.4),
         Inches(6), Inches(0.5), size=20, bold=True, color=ACCENT)
add_bullets(s, [
    "Contract_period 共線性 0.973 → 0.438",
    "XGBoost AUC 0.9796 → 0.9809（微升）",
    "K-means 群結構穩定",
    "特徵工程的價值：可信度 > accuracy",
], Inches(0.55), Inches(5.95), Inches(12), Inches(1.5), size=18)
add_page_number(s, 9)
add_footer_brand(s)
add_notes(s,
    "V1 的 Month_to_end_contract 跟 Contract_period 共線性 0.97，是病態值。"
    "替換成『已過合約月數』後共線性降到 0.44，模型表現持平甚至微升。"
    "這段是反思章節的關鍵金句：『特徵工程不只追求 accuracy，更是讓模型可信』。"
)

# ---------- Slide 9: K-means ----------
s = blank(prs)
add_title_bar(s, "客戶分群（STP · Segmentation）",
              "K-means K=4：4 群有顯著不同的流失率")
add_image(s, "figures/10_elbow.png",
          Inches(0.55), Inches(1.85), width=Inches(4))
add_image(s, "figures/11_kmeans_pca.png",
          Inches(4.8), Inches(1.85), width=Inches(4))
add_image(s, "figures/12_churn_per_cluster.png",
          Inches(9.05), Inches(1.85), width=Inches(3.8))

# V2 cluster profile 從 cluster_profile_v2.csv 讀來
add_text(s, "📋 V2 K-means 4 群（按 churn 排序）",
         Inches(0.55), Inches(5.4), Inches(12), Inches(0.5),
         size=18, bold=True, color=PRIMARY)
add_bullets(s, [
    "試水族  (n=1488, churn 54%)  · Lifetime 2.27 · Contract 2.36 · Freq 0.96/週",
    "熄火族  (n=384,  churn 27%)  · 100% 不留電話（隱私重視）",
    "活躍短期 (n=1063, churn 8%)   · Contract 8.32 月（實際最長）",
    "核心 VIP (n=1065, churn 6%)   · Freq 2.78/週（最高）",
], Inches(0.55), Inches(5.85), Inches(12.5), Inches(1.5), size=15)
add_page_number(s, 10)
add_footer_brand(s)
add_notes(s,
    "K=4 是 statistical 跟 business 的折衷選擇。"
    "Elbow / Silhouette 沒有絕對指向 K=4，但 K=4 在『group churn rate 差距』與『每群可設計策略』之間最 actionable。"
    "重要發現：兩個舊群名（活躍短期、熄火族）在 V2 K-means 的資料 signature 上其實名實不符 — 見 notebook §7.6 完整分析。"
)

# ---------- Slide 10: CLV ----------
s = blank(prs)
add_title_bar(s, "客戶終身價值 CLV（4P · Price / STP · Targeting）",
              "用 chocoZAP 台北費率（入會 500 + 月費 1,100）計算")
add_image(s, "figures/13_clv_per_cluster.png",
          Inches(0.55), Inches(1.85), width=Inches(6))
add_image(s, "figures/14_clv_vs_churn_scatter.png",
          Inches(6.85), Inches(1.85), width=Inches(6))

# 留客預算上限提示
add_text(s, "💰 留客預算上限 = CLV × 20%-30%",
         Inches(0.55), Inches(5.7), Inches(12), Inches(0.6),
         size=22, bold=True, color=ACCENT)
add_bullets(s, [
    "核心 VIP CLV 最高 → 投資 ROI 最好",
    "試水族 CLV 低（chocoZAP profile 1,758 NT）→ 不適合客製化教練",
    "→ chocoZAP 用「1 元入會降 CAC、無綁約降抗拒」應對",
], Inches(0.55), Inches(6.2), Inches(12.5), Inches(1.2), size=16)
add_page_number(s, 11)
add_footer_brand(s)
add_notes(s,
    "把抽象 Lifetime × 月費轉成台幣，行銷部就能直接算每群可投多少預算。"
    "Cluster 0 試水族 CLV 約 1,758 NT，行銷預算最多 NT$ 352 (20%)。"
    "這就是為什麼 chocoZAP 給的 offer 是低成本通用券，而不是教練客製。"
)

# ---------- Slide 11: 4 personas ----------
s = blank(prs)
add_title_bar(s, "4 人物誌（STP · Positioning）",
              "資料分群 → 真實人物 → 行銷可執行")

persona_cards = [
    ("試水族 / 我",   "22 歲 CS 大三",  "54% 流失",
     "1 元入會、5 月去 2 次、APP 約不到 KTV → 退會"),
    ("熄火族 / Eric", "31 歲金融",  "27% 流失",
     "新婚壓力大、頻率從 2/週→2/月、典型早期預警"),
    ("活躍短期 / 阿凱","27 歲房仲",  "8% 流失",
     "短約月月續、自學重訓、頻率 3/週、最划算用戶"),
    ("核心 VIP / Amy","38 歲 IT 主管", "6% 流失",
     "12 月長約、團體課常客、教練包月、5 年會員"),
]
y0 = 1.85
for i, (title, sub, churn, body) in enumerate(persona_cards):
    col = i % 2
    row = i // 2
    left = Inches(0.55 + col * 6.2)
    top  = Inches(y0 + row * 2.55)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               left, top, Inches(6), Inches(2.3))
    card.fill.solid(); card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = PRIMARY
    card.line.width = Pt(1)
    # title
    add_text(s, title, left + Inches(0.2), top + Inches(0.1),
             Inches(5.6), Inches(0.5), size=20, bold=True, color=ACCENT)
    add_text(s, f"{sub}  ·  {churn}", left + Inches(0.2), top + Inches(0.65),
             Inches(5.6), Inches(0.4), size=14, color=GREY)
    add_text(s, body, left + Inches(0.2), top + Inches(1.1),
             Inches(5.6), Inches(1.1), size=15, color=PRIMARY)
add_page_number(s, 12)
add_footer_brand(s)
add_notes(s,
    "4 個 persona 對應 4 個 cluster，行銷部可以對著名字設計訊息。"
    "其中 Cluster 0 (試水族) 完全是本作者的真實經驗，含『1 元入會、KTV 約不到』等具體細節 — "
    "完整故事在 personas.md。"
)

# ---------- Slide 12: chocoZAP case study ----------
s = blank(prs)
add_title_bar(s, "chocoZAP 個案研究（核心洞察）",
              "模型告訴你『誰會走』、訪談告訴你『為什麼走』")
# 矛盾流程
add_text(s, "🔁 chocoZAP 商業模式的內在矛盾",
         Inches(0.55), Inches(1.85), Inches(12), Inches(0.5),
         size=22, bold=True, color=ACCENT)
add_bullets(s, [
    "🎯 行銷端：用 KTV、按摩椅等稀缺設施吸客  →  抬高預期",
    "🏋️ 營運端：設施供給上限低（每店就那幾台）",
    "🔄 結果：新會員越多 → 預約越擠 → 失望越多 → 流失越快",
    "📊 量化驗證：模型預測 Cluster 0 流失 54%，本作者真實流失",
    "💡 chocoZAP 對策：不靠單客賺、靠量取勝（CAC 低 + 規模補）",
], Inches(0.55), Inches(2.4), Inches(12.5), Inches(3.5), size=18)

# 結論 callout
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                          Inches(0.55), Inches(5.7),
                          Inches(12.3), Inches(1.3))
box.fill.solid(); box.fill.fore_color.rgb = PRIMARY
box.line.fill.background()
add_text(s,
    "STP 鎖定 Cluster 0 試水族 → 接受高流失 → 靠商業模式（無綁約 + 1 元 CAC）取勝",
    Inches(0.7), Inches(5.95), Inches(12), Inches(0.5),
    size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s,
    "= 量化模型 + 質化訪談 + 商業洞察的三角驗證",
    Inches(0.7), Inches(6.45), Inches(12), Inches(0.5),
    size=14, color=WHITE, italic=True, align=PP_ALIGN.CENTER)
add_page_number(s, 13)
add_footer_brand(s)
add_notes(s,
    "這頁是整份報告最有商業洞察的一段："
    "我本人就是被 chocoZAP 命中又流失的真實樣本，"
    "不只是『客群結構問題』，更是 chocoZAP 自身的供需錯配。"
    "對應老師『競爭分析 2.0 = 文獻 + 訪談 + 情緒』框架。"
)

# ---------- Slide 13: 4P 商業建議 ----------
s = blank(prs)
add_title_bar(s, "商業建議（4P 總結）",
              "每群對應不同 Product / Price / Promotion 行動")

actions = [
    ("試水族",
     "Promotion: 30 天打卡挑戰、同儕邀請、預約優先",
     "Price: 1 元入會 + 升級長約折扣"),
    ("熄火族",
     "Promotion: 早期 SMS、免費團體課邀請",
     "避免群發續約信"),
    ("活躍短期",
     "Promotion: 跨分店通行、教練體驗包",
     "Price: 升級長約一次付 85 折"),
    ("核心 VIP",
     "Promotion: 推薦獎勵、年終回饋禮",
     "別做奇怪的事 — 不要打擾"),
]
y0 = 1.9
for i, (name, action1, action2) in enumerate(actions):
    top = Inches(y0 + i * 1.2)
    # color stripe
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(0.55), top,
                                 Inches(0.18), Inches(1))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = [ACCENT, RGBColor(0xFF, 0x98, 0x00),
                                    RGBColor(0xFF, 0xC1, 0x07),
                                    RGBColor(0x4C, 0xAF, 0x50)][i]
    stripe.line.fill.background()
    add_text(s, name, Inches(0.85), top,
             Inches(2.4), Inches(0.45), size=20, bold=True, color=PRIMARY)
    add_text(s, action1, Inches(3.3), top,
             Inches(9.5), Inches(0.45), size=15, color=PRIMARY)
    add_text(s, action2, Inches(3.3), top + Inches(0.5),
             Inches(9.5), Inches(0.45), size=15, color=GREY)
add_page_number(s, 14)
add_footer_brand(s)
add_notes(s,
    "每群行動清單來自 personas.md。"
    "重點是不同群該有不同 4P — 老師強調 STP 與 4P 是綁在一起的。"
)

# ---------- Slide 14: 反思 ----------
s = blank(prs)
add_title_bar(s, "反思與限制",
              "已知不足 + 未來工作")
# Left: 已解決 / 部分解決
add_text(s, "✅ 本研究已處理", Inches(0.55), Inches(1.85),
         Inches(6), Inches(0.5), size=20, bold=True,
         color=RGBColor(0x4C, 0xAF, 0x50))
add_bullets(s, [
    "Contract_period ↔ M2E 共線性 0.97 → 0.44 (V2)",
    "Cluster Profiling SOP 揭露兩個舊群名名實不符",
    "三方法分群對照（K-means / GMM / Hier）",
], Inches(0.55), Inches(2.4), Inches(6.2), Inches(3.5), size=16)

# Right: 限制
add_text(s, "⚠️ 仍存在的限制", Inches(7), Inches(1.85),
         Inches(6), Inches(0.5), size=20, bold=True, color=ACCENT)
add_bullets(s, [
    "freq_total ↔ current_month 仍共線（V2 未處理）",
    "缺 NPS / 滿意度資料 — 無法解釋『為什麼留下』",
    "Cluster 0 質性訪談 n=1（僅本作者）",
    "137 筆樣本時間欄位 rounding 不一致",
    "未做 uplift modeling / SMOTE 平衡校正",
    "CLV 用單一月費假設、未做 sensitivity",
    "資料為 Yandex Practicum 教學集（非真實營運）",
], Inches(7), Inches(2.4), Inches(6.2), Inches(4.5), size=14)
add_page_number(s, 15)
add_footer_brand(s)
add_notes(s,
    "這頁是給老師看『我知道自己的不足』。"
    "MDS 反思章節的標準寫法：先列已解決，再列待處理，避免讓人覺得自滿。"
)

# ---------- Divider 3 ----------
add_divider(prs, "第四部分", "Demo")

# ---------- Slide 15: Demo ----------
s = blank(prs)
add_title_bar(s, "Streamlit 互動 Demo (V2 模型)",
              "輸入會員特徵 → 即時預測流失機率、分群、CLV、行銷預算")

add_text(s, "▶ 啟動指令", Inches(0.55), Inches(1.85),
         Inches(6), Inches(0.5), size=20, bold=True, color=ACCENT)
# code box
code_box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(0.55), Inches(2.4),
                               Inches(6), Inches(1.4))
code_box.fill.solid(); code_box.fill.fore_color.rgb = LIGHT_BG
code_box.line.color.rgb = GREY
add_text(s,
    "cd C:\\Users\\cihci\\Desktop\\MDS\\gym_churn\n"
    ".\\.venv\\Scripts\\Activate.ps1\n"
    "streamlit run app.py\n"
    "→ http://localhost:8501",
    Inches(0.75), Inches(2.55), Inches(5.7), Inches(1.2),
    size=14, color=PRIMARY)

add_text(s, "✨ 互動功能", Inches(7), Inches(1.85),
         Inches(6), Inches(0.5), size=20, bold=True, color=ACCENT)
add_bullets(s, [
    "Sidebar 一鍵載入我的 chocoZAP profile",
    "切換分群方法（K-means / GMM / Hierarchical）",
    "GMM 顯示「屬於各群」的軟分群機率",
    "即時 SHAP / PCA / CLV 圖嵌入",
], Inches(7), Inches(2.4), Inches(6), Inches(3), size=15)

add_placeholder(s,
    "〔TODO：插入 Streamlit demo 截圖 1（chocoZAP profile 載入）〕",
    Inches(0.55), Inches(4.3), Inches(6), Inches(2.6))
add_placeholder(s,
    "〔TODO：插入 Streamlit demo 截圖 2（分群方法切換）〕",
    Inches(7), Inches(4.3), Inches(5.8), Inches(2.6))
add_page_number(s, 16)
add_footer_brand(s)
add_notes(s,
    "現場 demo 時：(1) 載入 chocoZAP profile → 看到流失 83%、試水族、CLV 1758 NT；"
    "(2) 切到 GMM 看軟分群機率；(3) 切到 Hierarchical 看不同分群結果。"
    "重點是『資料產品』的可操作感。"
)

# ---------- Slide 16: Q&A ----------
s = blank(prs)
# full bg
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
bg.fill.solid(); bg.fill.fore_color.rgb = PRIMARY
bg.line.fill.background()
# accent
add_text(s, "Q & A", Inches(0.5), Inches(2.3),
         Inches(12.3), Inches(1.5),
         size=96, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s,
    "謝謝聆聽 · 歡迎提問",
    Inches(0.5), Inches(4.2), Inches(12.3), Inches(0.6),
    size=24, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.CENTER)
add_text(s,
    "MDS 期末報告 · 健身房會員流失預測與留客策略 · chocoZAP 個案",
    Inches(0.5), Inches(5.0), Inches(12.3), Inches(0.5),
    size=14, color=RGBColor(0x99, 0x99, 0x99), italic=True, align=PP_ALIGN.CENTER)
add_page_number(s, 17)
add_notes(s,
    "結尾頁。常見可能問題：(1) 為什麼選 K=4 而不是 K=6？"
    "(2) chocoZAP 的數字怎麼來的？ (3) 模型可以即時上線嗎？"
)

# =====================================================================
prs.save(str(OUT))
print(f"✅ Saved {OUT.name}  ({OUT.stat().st_size // 1024} KB)")
print(f"   投影片總數：{len(prs.slides)}")
