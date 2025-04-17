import os
import json
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent, ImageSendMessage
from linebot.exceptions import LineBotApiError
from datetime import datetime, timedelta
import pytz
import aiohttp
import logging
import asyncio
from pathlib import Path
from typing import Dict, List

# 環境変数読み込み
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET      = os.getenv("LINE_CHANNEL_SECRET")
LINE_USER_ID             = os.getenv("LINE_USER_ID")
GROQ_API_KEY             = os.getenv("XAI_API_KEY")
VERCEL_DOMAIN            = os.getenv("VERCEL_DOMAIN", "https://your-vercel-app.vercel.app")

app = FastAPI()
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
parser       = WebhookParser(LINE_CHANNEL_SECRET)

USER_DATA_FILE = "user_data.json"
STATIC_DIR     = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────── MemoryManager ──────────────
class MemoryManager:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()

    def load_data(self) -> Dict[str, Dict]:
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_data(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_message(self, user_id: str, msg_type: str, text: str):
        entry = {"type": msg_type, "text": text, "timestamp": datetime.now().isoformat()}
        self.data.setdefault(user_id, {}).setdefault("messages", []).append(entry)
        self.save_data()

    def set_preference(self, user_id: str, key: str, value):
        self.data.setdefault(user_id, {}).setdefault("preferences", {})[key] = value
        self.save_data()

    def get_preference(self, user_id: str, key: str):
        return self.data.get(user_id, {}).get("preferences", {}).get(key)

    def check_image_quota(self, user_id: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        quota = self.data.setdefault(user_id, {}).setdefault("quota", {"date": today, "count": 0})
        if quota["date"] != today:
            quota["date"]  = today
            quota["count"] = 0
        if quota["count"] >= 1:
            return False
        quota["count"] += 1
        self.save_data()
        return True

memory_manager = MemoryManager(USER_DATA_FILE)

# ────────────── Grok 連携 ──────────────
async def ask_grok_mini(prompt: str, user_id: str) -> str:
    return f"簡易応答：{prompt}"

async def ask_grok_deepsearch(prompt: str, user_id: str, retries=3, delay=5) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "あなたは有能な株式分析執事AIです。礼儀正しく、簡潔に返答してください。"},
            {"role": "user",   "content": prompt}
        ]
    }

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("choices") and result["choices"][0]["message"].get("content"):
                            return result["choices"][0]["message"]["content"]
                        else:
                            return "申し訳ありません、AIからの返答がございませんでした。"
        except Exception as e:
            logger.error(f"Grok例外: {e}")
            await asyncio.sleep(delay)
    return await ask_grok_mini(prompt, user_id)

async def ask_grok_image(prompt: str, retries=3, delay=2) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "model": "grok-2-画像", "n": 1, "response_format": "url"}
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.x.ai/v1/images/generations", headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["data"][0]["url"]
        except Exception as e:
            logger.error(f"画像API例外: {e}")
            await asyncio.sleep(delay)
    return None

# ────────────── カスタム予測 ──────────────
async def custom_tech(prompt: str, user_id: str) -> str:
    ticker = memory_manager.get_preference(user_id, "favorite_stock") or "7203.T"
    try:
        data = yf.download(ticker, period="1mo")
        last_price = data['Close'][-1]
        last_5 = data['Close'][-5:]
        changes = [(last_5[i] - last_5[i-1]) / last_5[i-1] for i in range(1, len(last_5))]
        avg_change = sum(changes) / len(changes)
        predicted = last_price * (1 + avg_change)
        return f"{ticker} の翌日予測株価: {predicted:.2f}円（平均変化率: {avg_change*100:.2f}%）"
    except Exception as e:
        logger.error(f"予測エラー: {e}")
        return await ask_grok_mini(f"{ticker} の株価予測を簡潔に教えて", user_id)

# ────────────── チャート生成 ──────────────
def generate_chart_image(ticker: str) -> Path:
    try:
        data = yf.download(ticker, period="1mo")
        plt.figure(figsize=(8, 4))
        plt.plot(data['Close'], label=ticker)
        plt.title(f"{ticker} 株価推移（1ヶ月）")
        plt.xlabel("日付")
        plt.ylabel("株価")
        plt.grid(True)
        plt.tight_layout()
        path = STATIC_DIR / f"{ticker}_chart.png"
        plt.savefig(path)
        plt.close()
        return path
    except Exception as e:
        logger.error(f"チャート生成エラー: {e}")
        return None

# ────────────── LINE Webhook ──────────────
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body      = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        # 友だち追加登録
        if isinstance(event, FollowEvent):
            uid = event.source.user_id
            memory_manager.add_message(uid, "フォロー", "ご登録ありがとうございます！")
            try:
                line_bot_api.push_message(uid, TextSendMessage(text="ご登録ありがとうございます、旦那様！"))
            except LineBotApiError as e:
                logger.error(f"FollowEventエラー: {e}")

        # メッセージ処理
        elif isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
            uid  = event.source.user_id
            text = event.message.text.strip()
            memory_manager.add_message(uid, "受信", text)

            # おすすめ株
            if any(w in text for w in ["株", "おすすめ", "買い時", "下がった"]):
                stock = memory_manager.get_preference(uid, "favorite_stock") or "7203.T"
                df    = yf.Ticker(stock).history(period="1mo")
                price = df['Close'][-1]
                reply = f"おすすめ銘柄: {stock}\n株価: {price:.2f}円"
            # 調べてコマンド
            elif text.upper().startswith("調べて"):
                ticker = text.upper().replace("調べて", "").strip()
                df     = yf.Ticker(ticker).history(period="1mo")
                price  = df['Close'][-1]
                reply = f"{ticker} の株価: {price:.2f}円"
                memory_manager.set_preference(uid, "favorite_stock", ticker)
            # 分析コマンド
            elif text.startswith("分析"):
                query = text.replace("分析", "").strip()
                reply = await ask_grok_deepsearch(query, uid)
            # 検索コマンド
            elif text.startswith("検索"):
                query = text.replace("検索", "").strip()
                reply = await ask_grok_deepsearch(query, uid)
            # 画像生成コマンド
            elif text.startswith("画像"):
                if not memory_manager.check_image_quota(uid):
                    reply = "本日の画像生成クォータを使い切りました。"
                else:
                    query = text.replace("画像", "").strip()
                    url   = await ask_grok_image(query)
                    if url:
                        memory_manager.add_message(uid, "送信画像", url)
                        line_bot_api.reply_message(
                            event.reply_token,
                            ImageSendMessage(original_content_url=url, preview_image_url=url)
                        )
                        continue
                    else:
                        reply = "画像生成に失敗しました。"
            # カスタム予測
            elif text.startswith("カスタム"):
                query = text.replace("カスタム", "").strip()
                reply = await custom_tech(query, uid)
            else:
                reply = "ご用命があれば、いつでもお申し付けくださいませ。"

            # 返信実行
            try:
                line_bot_api.reply_message(uid and event.reply_token, TextSendMessage(text=reply))
            except LineBotApiError as e:
                logger.error(f"MessageEventエラー: {e}")

    return "OK"

# ────────────── CSVエクスポート ──────────────
@app.get("/export-csv")
async def export_csv():
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ユーザーID", "種類", "内容", "タイムスタンプ"])

    for uid, info in memory_manager.data.items():
        for msg in info.get("messages", []):
            writer.writerow([uid, msg["type"], msg["text"], msg["timestamp"]])

    return {"filename": "log.csv", "content": output.getvalue()}

# ────────────── RSIアラート通知 ──────────────
@app.get("/rsi-alert")
async def rsi_alert():
    tickers = ["7203.T", "6758.T"]
    for ticker in tickers:
        df  = yf.download(ticker, period="1mo")
        rsi = calc_rsi(df['Close'])
        if rsi < 25:
            for uid in memory_manager.data:
                line_bot_api.push_message(
                    uid,
                    TextSendMessage(text=f"⚠️ アラート: {ticker} のRSIが{rsi:.2f}です。")
                )
    return {"message": "アラートチェック完了"}

# ────────────── 管理ダッシュボード ──────────────
@app.get("/admin")
async def admin_dashboard():
    total_users    = len(memory_manager.data)
    total_messages = sum(len(info.get("messages", [])) for info in memory_manager.data.values())
    return {
        "総ユーザー数": total_users,
        "総メッセージ数": total_messages,
        "最終更新": datetime.now().isoformat()
    }

# ────────────── モーニングレポート ──────────────
@app.get("/morning-report")
async def morning_report():
    tokyo = pytz.timezone("Asia/Tokyo")
    now   = datetime.now(tokyo)
    if now.hour != 9:
        return {"message": "9時以外は実行されません"}
    # 注目銘柄
    stock  = memory_manager.get_preference(LINE_USER_ID, "favorite_stock") or "7203.T"
    df     = yf.Ticker(stock).history(period="1mo")
    price  = df['Close'][-1]
    msg    = f"おはようございます、旦那様！\n注目銘柄: {stock}\n株価: {price:.2f}円"
    # チャート画像
    path   = generate_chart_image(stock)
    chart_url = f"{VERCEL_DOMAIN}/static/{path.name}" if path else None

    for uid in memory_manager.data:
        line_bot_api.push_message(uid, TextSendMessage(text=msg))
        if chart_url:
            line_bot_api.push_message(
                uid,
                ImageSendMessage(original_content_url=chart_url, preview_image_url=chart_url)
            )
    return {"message": "モーニングレポート送信完了"}

# ────────────── テスト送信 ──────────────
@app.get("/test-message")
async def test_message():
    if not LINE_USER_ID:
        return {"message": "LINE_USER_IDが未設定です"}
    line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text="テストメッセージでございます！"))
    return {"message": f"{LINE_USER_ID} にテスト送信完了"}
