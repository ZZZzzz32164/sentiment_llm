# -*- coding: utf-8 -*-
"""预测封装：加载训练好的模型与向量化器，输入文本即可判别情感。"""
import os
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model.joblib")
VEC_PATH = os.path.join(ROOT, "vectorizer.joblib")

_LABEL_MAP = {1: "正向", 0: "负向"}


def load():
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    return model, vec


def predict_one(text: str) -> str:
    """判别单条文本情感，返回"正向"/"负向"。"""
    model, vec = load()
    X = vec.transform([text])
    lab = int(model.predict(X)[0])
    return _LABEL_MAP.get(lab, "未知")


def predict_many(texts):
    model, vec = load()
    X = vec.transform(texts)
    labels = model.predict(X)
    return [_LABEL_MAP.get(int(l), "未知") for l in labels]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_one(sys.argv[1]))
    else:
        for s in ["这家酒店服务很好，房间干净舒适。", "房间太差，前台态度也不好。"]:
            print(repr(s), "->", predict_one(s))
