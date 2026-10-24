# -*- coding: utf-8 -*-
"""数据加载与清洗：读取公开情感数据集，Pandas 去重、缺失值处理。"""
import os
import numpy as np
import pandas as pd

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data.csv")


def load_data(path: str = None) -> pd.DataFrame:
    """读取公开中文情感语料，返回 (label, review) 两列 DataFrame。"""
    if path is None:
        path = DEFAULT_PATH
    df = pd.read_csv(path)

    # 只保留模型需要的两列（online_shopping_10_cats 含额外 cat 类目列，一并丢弃）
    keep = [c for c in ("label", "review") if c in df.columns]
    df = df[keep]

    # 统一字段
    df["review"] = df["review"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # 缺失值处理
    df = df.dropna(subset=["review"])
    df = df[df["review"] != ""]

    # 去重
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    return df


def split_sets(df: pd.DataFrame, n_train: int = 300, n_eval: int = 200,
               seed: int = 42):
    """把语料拆成【LLM 标注训练池】与【有真标签评估集】。
    - train_pool：视为"无标注"，交给大模型批量标注后作为训练集；
    - eval_set：保留真实标签，用于对比朴素贝叶斯与大模型的判别准确率。
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(df.index).tolist()
    train_idx = idx[:n_train]
    eval_idx = idx[n_train:n_train + n_eval]
    train_pool = df.loc[train_idx].reset_index(drop=True)
    eval_set = df.loc[eval_idx].reset_index(drop=True)
    return train_pool, eval_set


if __name__ == "__main__":
    d = load_data()
    print("总样本数:", len(d))
    print("标签分布:\n", d["label"].value_counts())
