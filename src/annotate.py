# -*- coding: utf-8 -*-
"""大模型标注：调用 DeepSeek API（HTTP+JSON）对无标注评论批量生成情感标签。"""
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

PROMPT = (
    "你是情感分析助手。判断下面这条中文评论的情感是“正向”还是“负向”。\n"
    "只回答两个字：正向 或 负向，不要输出其他内容。\n\n评论：{}"
)


def llm_annotate(review: str, retries: int = 3) -> int:
    """单条评论调用大模型，返回情感标签：1=正向，0=负向；失败返回 None。"""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(review)}],
        "temperature": 0,
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            # 稳健解析：含"负"且不含"正" → 负向；含"正" → 正向
            if "负" in ans and "正" not in ans:
                return 0
            if "正" in ans:
                return 1
            return None
        except Exception as e:
            print("  [retry] 调用失败:", type(e).__name__, str(e)[:80])
            time.sleep(2 * (i + 1))
    return None


def annotate(df: pd.DataFrame, out_path: str = None, log_every: int = 25) -> pd.DataFrame:
    """对 df 的 review 列逐条调用大模型标注，返回带 llm_label 列的新 DataFrame。"""
    reviews = df["review"].tolist()
    labels = []
    total = len(reviews)
    for i, rv in enumerate(reviews):
        lab = llm_annotate(rv)
        labels.append(lab)
        if (i + 1) % log_every == 0:
            print(f"  已标注 {i + 1}/{total}")
    out = df.copy()
    out["llm_label"] = labels
    if out_path:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out
