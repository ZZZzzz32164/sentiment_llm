# -*- coding: utf-8 -*-
"""NLP 特征工程：jieba 中文分词 + 停用词过滤 + TF-IDF 向量化。"""
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 内置常用中文停用词（精简版）
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
    "他们", "这", "那", "有", "也", "就", "都", "而", "及", "与", "或", "一个",
    "个", "很", "太", "非常", "比较", "还", "又", "再", "但", "但是", "因为",
    "所以", "如果", "虽然", "以及", "之", "其", "为", "到", "从", "把", "被",
    "对", "于", "给", "让", "使", "等", "等?", "?", "。", "，", "！", "？", ".",
    ",", "!", "?", "、", "；", "：", "\"", "”", "“", "'", "（", "）", "(", ")"
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+")


def clean(text: str) -> str:
    """基础清洗：去链接、去数字串、去空白。"""
    text = _URL_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """jieba 分词 + 去停用词，返回词列表。"""
    text = clean(text)
    words = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w or w in STOPWORDS or len(w) == 1:
            continue
        words.append(w)
    return words


def build_vectorizer():
    """构建 TF-IDF 向量化器，使用 jieba 分词器。"""
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )


def fit_vectorizer(corpus):
    vec = build_vectorizer()
    X = vec.fit_transform(corpus)
    return vec, X
