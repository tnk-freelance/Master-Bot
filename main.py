mport os
import json
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent
from linebot.exceptions import LineBotApiError
from datetime import datetime
import pytz
import aiohttp
import logging
import asyncio
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
        if quota["count"] >= 1:
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
                        return result["choices"][0]["message"]["content
