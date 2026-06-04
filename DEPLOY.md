# Streamlit Cloud 部署指南

從 0 到拿到公開網址，**約 15-20 分鐘**。

---

## 步驟 0：準備工作（一次性）

### 0.1 確認 Git 已裝
新開 PowerShell（讓 PATH 重新載入）執行：
```powershell
git --version
```
看到 `git version 2.xx.x` 就 OK。

### 0.2 設定 Git 身分
**第一次用 Git 才需要**。用你的 GitHub email：
```powershell
git config --global user.name "你的名字"
git config --global user.email "erjunsheng@gmail.com"
```

### 0.3 在 GitHub 建一個 repo
1. 打開 https://github.com/new
2. Repository name 填：`gym-churn-streamlit`（或你喜歡的名字）
3. 選 **Public**（Streamlit Cloud 免費方案需要公開 repo；如要私有可改 Private，免費版也支援 1 個私有 app）
4. **不要**勾「Add a README」、「Add .gitignore」（我們會從本機 push 上去）
5. 點 **Create repository**
6. 看到 GitHub 給你一個網址類似：`https://github.com/你的帳號/gym-churn-streamlit.git` — 複製起來

---

## 步驟 1：初始化 + 上傳到 GitHub

打開 PowerShell，跑：

```powershell
cd C:\Users\cihci\Desktop\MDS\gym_churn

# 初始化 git repo
git init

# 設定預設分支為 main
git branch -M main

# 加入所有檔案（.gitignore 會自動排除 .venv、__pycache__ 等）
git add .

# 第一次 commit
git commit -m "Initial commit: gym churn V2 + 3 clustering methods"

# 連到你 GitHub 上的 repo（換成你的網址）
git remote add origin https://github.com/你的帳號/gym-churn-streamlit.git

# 推上去
git push -u origin main
```

> **第一次 push 會跳 GitHub 登入視窗**，用你的 GitHub 帳號登入授權。
> 若你之前沒設過 token，會被要求建立 Personal Access Token：
> 1. 跟著畫面去 https://github.com/settings/tokens
> 2. Generate new token (classic) → 勾 `repo` 權限 → Generate
> 3. 複製 token 貼回 PowerShell 當密碼

---

## 步驟 2：部署到 Streamlit Cloud

1. 打開 https://share.streamlit.io
2. 點右上 **Sign in with GitHub**（用你的 GitHub 登入）
3. 第一次登入會跳授權，按 **Authorize Streamlit**
4. 進到 dashboard 點 **New app**（右上）
5. 填表單：
   - **Repository**：選 `你的帳號/gym-churn-streamlit`
   - **Branch**：`main`
   - **Main file path**：`app.py`
   - **App URL**（自定）：例如 `gym-churn-mds`（會變成 `gym-churn-mds.streamlit.app`）
6. 點 **Deploy!**
7. 等 5-10 分鐘看到「Your app is live!」

---

## 步驟 3：拿到公開網址

部署成功後你會看到：
```
https://gym-churn-mds.streamlit.app
```

→ **這就是你的 demo 網址，任何電腦都能打開**。

---

## 步驟 4（強烈建議）：放進簡報

### 4.1 把網址寫進 Slide 16

打開 `MDS_期末報告.pptx` → Slide 16（Demo 頁）→ 把目前的 localhost 網址改成你的公開網址。

### 4.2 產生 QR code 讓老師現場掃

最快方法：
1. 去 https://www.qrcode-monkey.com（或任一免費 QR 產生器）
2. 貼上你的網址
3. 下載 PNG
4. 拖到 Slide 16 的截圖佔位框

或用 Python 自己產：
```powershell
.\.venv\Scripts\python.exe -c "import qrcode; qrcode.make('https://你的網址.streamlit.app').save('qr.png')"
```
（需先 `pip install qrcode[pil]`）

---

## 之後改 code 怎麼更新？

```powershell
git add .
git commit -m "說明這次改了什麼"
git push
```

**Streamlit Cloud 會自動偵測 push 並重 deploy**（約 2-3 分鐘）。不用再進 dashboard。

---

## 常見錯誤排解

| 症狀 | 解法 |
|---|---|
| `git: command not found` | 開新 PowerShell（PATH 才會生效）；或重啟電腦 |
| Streamlit Cloud 卡在 "Installing dependencies" 超過 15 分鐘 | 看 logs；通常是 requirements.txt 某套件版本不存在 |
| App 跑起來但顯示 "ModuleNotFoundError" | 漏裝套件，補進 `requirements.txt` 再 push |
| App 跑起來但顯示 "FileNotFoundError: xgb_churn_model_v2.pkl" | 沒 push 上去；確認 `.gitignore` 沒擋到，重新 `git add -f xgb_churn_model_v2.pkl` |
| Pickle 反序列化失敗 | 本機 sklearn / xgboost 版本跟雲端對不上，鎖更緊的 version |
| App 跑超慢 / 出現 1GB RAM 限制 | 把不必要的圖片從 figures_v2/ 拿掉再 push |

---

## 上 Streamlit Cloud 後會帶上去的檔案清單

✅ 會 push（app 需要的）：
- `app.py`
- `requirements.txt`（精簡版）
- `xgb_churn_model_v2.pkl` ~289 KB
- `clustering_models.pkl` ~98 KB
- `figures_v2/` 內 10 張 PNG ~1.9 MB
- `personas.md`（純文字，~11 KB）

❌ 不會 push（已被 .gitignore 排除）：
- `.venv/`（本機 Python 環境）
- `xgb_churn_model.pkl`（V1 模型不需要）
- `MDS_期末報告.pptx`（簡報）
- `analysis.ipynb`（notebook，1.7 MB）
- 所有 `trace_*.py`、`experiment_*.py`、`make_*.py`、`build_pptx.py`（開發腳本）
- `data/cluster_profile.csv`（V1 統計，app 不用）

→ **預估 push 上去總大小：~2.5 MB**。

---

## 給你的明確 next steps

1. ✅ **這份指南建好**（檔案：`DEPLOY.md`）
2. ✅ **.gitignore 已建**
3. ✅ **slim requirements.txt 已建**（舊版備份為 requirements_dev.txt）
4. ⏳ **等 git 安裝完成** → 跑步驟 1 的指令
5. ⏳ **GitHub 建 repo**（步驟 0.3）
6. ⏳ **Push**（步驟 1）
7. ⏳ **Streamlit Cloud deploy**（步驟 2）
8. ⏳ **網址 / QR code 進 PPT**（步驟 4）

預計 15-20 分鐘從這份指南到拿到公開網址。

任何卡關直接告訴我訊息，我幫你 debug。
