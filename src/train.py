# -*- coding: utf-8 -*-
"""模型训练与评估：朴素贝叶斯（jieba+TF-IDF），与大模型(DeepSeek)判别对比。"""
import os
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from data_loader import load_data, split_sets
from annotate import annotate, llm_annotate
from features import fit_vectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def train_and_evaluate(n_train: int = 300, n_eval: int = 200, seed: int = 42):
    """完整训练评估：LLM 标注训练集 → 朴素贝叶斯 → 与 LLM 准确率对比。"""
    df = load_data()
    train_pool, eval_set = split_sets(df, n_train=n_train, n_eval=n_eval, seed=seed)

    print("=" * 60)
    print("[1/4] 大模型标注训练池（DeepSeek，{} 条）...".format(n_train))
    ann_path = os.path.join(DATA_DIR, "llm_annotated.csv")
    annotated = annotate(train_pool, out_path=ann_path)

    # 用大模型标注的标签做训练集（丢弃标注失败的行）
    train_df = annotated.dropna(subset=["llm_label"]).copy()
    train_df["label"] = train_df["llm_label"].astype(int)
    print("   标注成功行:", len(train_df), "/", len(annotated))

    print("[2/4] 大模型对评估集判别的准确率（基准）...")
    eval_labels = eval_set["label"].astype(int).tolist()
    llm_eval_labels = [llm_annotate(rv) for rv in eval_set["review"].tolist()]
    llm_ok = [(a == b) for a, b in zip(llm_eval_labels, eval_labels) if a is not None]
    llm_acc = sum(llm_ok) / len(llm_ok) if llm_ok else 0.0
    print("   DeepSeek 评估集准确率: {:.4f} ({} 条有效)".format(llm_acc, len(llm_ok)))

    print("[3/4] jieba + TF-IDF 特征 + 朴素贝叶斯训练...")
    vec, X_train = fit_vectorizer(train_df["review"].tolist())
    model = MultinomialNB(alpha=1.0)
    cv = cross_val_score(model, X_train, train_df["label"].values, cv=5,
                         scoring="accuracy")
    print("   训练集 5 折交叉验证准确率: {:.4f} +/- {:.4f}".format(cv.mean(), cv.std()))
    model.fit(X_train, train_df["label"].values)

    print("[4/4] 朴素贝叶斯在评估集上的准确率（与大模型对比）...")
    X_eval = vec.transform(eval_set["review"].tolist())
    nb_pred = model.predict(X_eval)
    nb_acc = accuracy_score(eval_set["label"].values, nb_pred)
    print("   朴素贝叶斯评估集准确率: {:.4f}".format(nb_acc))
    print("=" * 60)
    print("对比结果  |  朴素贝叶斯: {:.4f}  |  DeepSeek 大模型: {:.4f}".format(nb_acc, llm_acc))

    # 保存产物
    model_path = os.path.join(ROOT, "model.joblib")
    vec_path = os.path.join(ROOT, "vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vec, vec_path)
    pd.DataFrame({
        "split": ["train_pool", "eval_set"],
        "n": [len(train_df), len(eval_set)],
        "nb_acc": [cv.mean(), nb_acc],
        "llm_acc": [llm_acc, llm_acc],
    }).to_csv(os.path.join(DATA_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    print("已保存: model.joblib / vectorizer.joblib / metrics.csv")


if __name__ == "__main__":
    train_and_evaluate()
