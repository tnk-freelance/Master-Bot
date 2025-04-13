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

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 環境変数
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VERCEL_DOMAIN = os.environ.get("VERCEL_DOMAIN", "https://masterbot-lime.vercel.app")

if not all([LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, GROQ_API_KEY]):
    logger.error("環境変数が不足しています")
    raise ValueError("環境変数が設定されていません")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(LINE_CHANNEL_SECRET)
USER_DATA_FILE = "user_data.json"
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# RAG風メモリ管理
class MemoryManager:
    def __init__(self):
        self.data: Dict[str, Dict] = self.load_data()

    def load_data(self) -> Dict:
        try:
            if os.path.exists(USER_DATA_FILE):
                with open(USER_DATA_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
        return {}

    def save_data(self):
        try:
            with open(USER_DATA_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"データ保存エラー: {e}")

    def add_message(self, user_id: str, message: str, response: str):
        if user_id not in self.data:
            self.data[user_id] = {"history": [], "preferences": {}, "image_quota": {"date": "", "count": 0}}
        self.data[user_id]["history"].append({"user": message, "response": response})
        self.data[user_id]["history"] = self.data[user_id]["history"][-10:]
        self.save_data()

    def get_context(self, user_id: str) -> List[Dict]:
        return self.data.get(user_id, {}).get("history", [])

    def set_preference(self, user_id: str, key: str, value: str):
        if user_id not in self.data:
            self.data[user_id] = {"history": [], "preferences": {}, "image_quota": {"date": "", "count": 0}}
        self.data[user_id]["preferences"][key] = value
        self.save_data()

    def get_preference(self, user_id: str, key: str) -> str:
        return self.data.get(user_id, {}).get("preferences", {}).get(key, "")

    def check_image_quota(self, user_id: str) -> bool:
        if user_id not in self.data:
            self.data[user_id] = {"history": [], "preferences": {}, "image_quota": {"date": "", "count": 0}}
        today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
        quota = self.data[user_id]["image_quota"]
        if quota["date"] != today:
            quota["date"] = today
            quota["count"] = 0
        if quota["count"] >= 1:  # 1日1回
            return False
        quota["count"] += 1
        self.save_data()
        return True

memory_manager = MemoryManager()

# N言語風処理
def process_context(text: str, user_id: str) -> str:
    context = memory_manager.get_context(user_id)
    context_str = "\n".join([f"User: {m['user']}\nBot: {m['response']}" for m in context[-3:]])
    return f"<CTX>{context_str}\nCurrent: {text}</CTX>"

def analyze_emotion(text: str) -> str:
    if any(word in text.lower() for word in ["嬉しい", "最高", "やった"]):
        return "<EMO>happy</EMO>"
    elif any(word in text.lower() for word in ["悲しい", "大変", "困った"]):
        return "<EMO>sad</EMO>"
    return "<EMO>neutral</EMO>"

# Grok API呼び出し（軽量）
async def ask_grok_mini(prompt: str, user_id: str, retries=3, delay=2) -> str:
    processed_prompt = process_context(prompt, user_id) + analyze_emotion(prompt)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "あなたは株式分析の執事です。簡潔に応答してください。"},
            {"role": "user", "content": processed_prompt}
        ],
        "model": "grok-3-mini",
        "temperature": 0.7
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await resp.text()
                        logger.error(f"Mini APIエラー: 試行 {attempt+1}/{retries}, ステータス: {resp.status}, 詳細: {error_text}")
                        if resp.status == 429:
                            await asyncio.sleep(delay * (2 ** attempt))
                        elif resp.status >= 500:
                            await asyncio.sleep(delay)
                        else:
                            break
        except Exception as e:
            logger.error(f"Mini API例外: 試行 {attempt+1}/{retries}, エラー: {e}")
            await asyncio.sleep(delay)
    return "申し訳ございません、Grok（Mini）との通信に失敗しました。"

# Grok API呼び出し（DeepSearch）
async def ask_grok_deepsearch(prompt: str, user_id: str, retries=3, delay=5) -> str:
    processed_prompt = process_context(prompt, user_id) + analyze_emotion(prompt)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "X-DeepSearch": "true"
    }
    data = {
        "messages": [
            {"role": "system", "content": "あなたは株式分析の執事です。最新情報をWebから取得し、丁寧に応答してください。"},
            {"role": "user", "content": processed_prompt}
        ],
        "model": "grok-3-速い",
        "temperature": 0.7,
        "deep_search": True
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await resp.text()
                        logger.error(f"DeepSearch APIエラー: 試行 {attempt+1}/{retries}, ステータス: {resp.status}, 詳細: {error_text}")
                        if resp.status == 429:
                            await asyncio.sleep(delay * (2 ** attempt))
                        elif resp.status >= 500:
                            await asyncio.sleep(delay)
                        else:
                            break
        except Exception as e:
            logger.error(f"DeepSearch API例外: 試行 {attempt+1}/{retries}, エラー: {e}")
            await asyncio.sleep(delay)
    return await ask_grok_mini(prompt, user_id)  # フォールバック

# Grok API呼び出し（画像生成）
async def ask_grok_image(prompt: str, retries=3, delay=2) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": "grok-2-画像",
        "n": 1,
        "response_format": "url"
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.x.ai/v1/images/generations", headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["data"][0]["url"]
                    else:
                        error_text = await resp.text()
                        logger.error(f"画像APIエラー: 試行 {attempt+1}/{retries}, ステータス: {resp.status}, 詳細: {error_text}")
                        if resp.status == 429:
                            await asyncio.sleep(delay * (2 ** attempt))
                        elif resp.status >= 500:
                            await asyncio.sleep(delay)
                        else:
                            break
        except Exception as e:
            logger.error(f"画像API例外: 試行 {attempt+1}/{retries}, エラー: {e}")
            await asyncio.sleep(delay)
    return None

# カスタム技術（仮：株価予測）
async def custom_tech(prompt: str, user_id: str) -> str:
    ticker = memory_manager.get_preference(user_id, "favorite_stock") or "7203.T"
    try:
        data = yf.download(ticker, period="1mo")
        last_price = data['Close'][-1]
        # 簡易予測（直近5日の平均変化率で翌日を予測）
        last_5_days = data['Close'][-5:]
        changes = [(last_5_days[i] - last_5_days[i-1]) / last_5_days[i-1] for i in range(1, len(last_5_days))]
        avg_change = sum(changes) / len(changes)
        predicted_price = last_price * (1 + avg_change)
        return f"{ticker} の翌日予測株価: {predicted_price:.2f}円（直近5日の平均変化率: {avg_change*100:.2f}%）"
    except Exception as e:
        logger.error(f"カスタム予測エラー: {e}")
        return await ask_grok_mini(f"{ticker} の株価予測を簡潔に教えて", user_id)

# チャート生成
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
        image_path = STATIC_DIR / f"{ticker}_chart.png"
        plt.savefig(image_path)
        plt.close()
        return image_path
    except Exception as e:
        logger.error(f"チャート生成エラー: {e}")
        return None

# LINE Webhook
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
    except Exception as e:
        logger.error(f"Webhookエラー: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, FollowEvent):
            uid = event.source.user_id
            memory_manager.add_message(uid, "フォロー", "ご登録ありがとうございます！")
            try:
                line_bot_api.push_message(uid, TextSendMessage(text="ご登録ありがとうございます、旦那様！執事がお相手いたします。"))
            except LineBotApiError as e:
                logger.error(f"FollowEventエラー: {e}")
        elif isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
            uid = event.source.user_id
            text = event.message.text.strip()
            reply = "ご用命があれば、いつでもお申し付けください。"
            
            if "株" in text or "おすすめ" in text:
                stock = memory_manager.get_preference(uid, "favorite_stock") or "7203.T"
                ticker = yf.Ticker(stock)
                reply = f"おすすめ銘柄: {stock}\n株価: {ticker.info.get('regularMarketPrice', '取得失敗')}円"
                memory_manager.add_message(uid, text, reply)
            elif text.startswith("調べて"):
                ticker = text.replace("調べて", "").strip().upper()
                if ticker.endswith(".T"):
                    stock = yf.Ticker(ticker)
                    reply = f"{ticker} の株価: {stock.info.get('regularMarketPrice', '取得失敗')}円"
                    memory_manager.set_preference(uid, "favorite_stock", ticker)
                    memory_manager.add_message(uid, text, reply)
                else:
                    reply = "日本株（例: 7203.T）を指定してください。"
                    memory_manager.add_message(uid, text, reply)
            elif text.startswith("分析"):
                query = text.replace("分析", "").strip()
                reply = await ask_grok_mini(query, uid)
                memory_manager.add_message(uid, text, reply)
            elif text.startswith("検索"):
                query = text.replace("検索", "").strip()
                reply = await ask_grok_deepsearch(query, uid)
                memory_manager.add_message(uid, text, reply)
            elif text.startswith("画像"):
                if not memory_manager.check_image_quota(uid):
                    reply = "本日の画像生成クォータ（1回）を達成しました。明日またお試しください。"
                    memory_manager.add_message(uid, text, reply)
                else:
                    query = text.replace("画像", "").strip()
                    image_url = await ask_grok_image(query)
                    if image_url:
                        try:
                            line_bot_api.reply_message(
                                event.reply_token,
                                ImageSendMessage(
                                    original_content_url=image_url,
                                    preview_image_url=image_url
                                )
                            )
                            memory_manager.add_message(uid, text, "画像を送信しました")
                            return "OK"
                        except LineBotApiError as e:
                            logger.error(f"画像送信エラー: {e}")
                            reply = "画像の送信に失敗しました。"
                            memory_manager.add_message(uid, text, reply)
                    else:
                        reply = "画像生成に失敗しました。もう一度お試しください。"
                        memory_manager.add_message(uid, text, reply)
            elif text.startswith("カスタム"):
                query = text.replace("カスタム", "").strip()
                reply = await custom_tech(query, uid)
                memory_manager.add_message(uid, text, reply)
            
            try:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            except LineBotApiError as e:
                logger.error(f"MessageEventエラー: {e}")
    return "OK"

# モーニングレポート
@app.get("/morning-report")
async def morning_report():
    try:
        tokyo = pytz.timezone("Asia/Tokyo")
        now = datetime.now(tokyo)
        if now.hour != 9:
            return {"message": "モーニングレポートは9時のみ実行されます"}
        
        stock = "7203.T"
        ticker = yf.Ticker(stock)
        message = (
            f"おはようございます、旦那様！\n"
            f"本日の注目銘柄: {stock}\n"
            f"株価: {ticker.info.get('regularMarketPrice', '取得失敗')}円"
        )
        chart_path = generate_chart_image(stock)
        chart_url = f"{VERCEL_DOMAIN}/static/{chart_path.name}" if chart_path else None
        
        for uid in memory_manager.data.keys():
            try:
                line_bot_api.push_message(uid, TextSendMessage(text=message))
                if chart_url:
                    line_bot_api.push_message(uid, ImageSendMessage(
                        original_content_url=chart_url,
                        preview_image_url=chart_url
                    ))
                memory_manager.add_message(uid, "モーニングレポート", message)
            except LineBotApiError as e:
                logger.error(f"レポート送信エラー (UID: {uid}): {e}")
        return {"message": "モーニングレポートを送信しました"}
    except Exception as e:
        logger.error(f"モーニングレポートエラー: {e}")
        return {"message": "レポート送信に失敗しました"}

# テスト送信
@app.get("/test-message")
async def test_message():
    try:
        user_id = os.environ.get("LINE_USER_ID")
        if not user_id:
            logger.error("LINE_USER_IDが未設定")
            return {"message": "LINE_USER_IDが未設定です"}
        line_bot_api.push_message(user_id, TextSendMessage(text="テストメッセージでございます！"))
        memory_manager.add_message(user_id, "テスト", "テストメッセージを送信")
        return {"message": f"ユーザー {user_id} にテスト送信しました"}
    except LineBotApiError as e:
        logger.error(f"テスト送信エラー: {e}")
        return {"message": f"テスト送信に失敗しました: {e}"}
