# 基于大模型的大规模文本情感分析系统（NLP+LLM）

公开中文情感语料 → **DeepSeek 大模型弱监督标注** → jieba + TF-IDF + 朴素贝叶斯 → 与大模型判别结果对比。输入任意中文文本，即可判别正向 / 负向情感。

## 技术栈
- 数据处理：Pandas（清洗去重、缺失值处理）
- 大模型标注：DeepSeek API（HTTP+JSON），弱监督批量生成情感标签
- NLP 特征：jieba 中文分词 + 停用词过滤 + TF-IDF
- 模型：scikit-learn 朴素贝叶斯（训练 / 交叉验证 / 评估）
- 工程：Git 版本管理，模块化（采集-标注-训练-预测），支持增量重训

## 项目结构
```
sentiment_llm/
├─ data/                  # 公开数据集 data.csv + 大模型标注结果 llm_annotated.csv + 指标 metrics.csv
├─ src/
│  ├─ data_loader.py      # 读取 / 清洗 / 拆分（Pandas）
│  ├─ annotate.py         # DeepSeek 大模型标注
│  ├─ features.py         # jieba 分词 + TF-IDF
│  ├─ train.py            # 朴素贝叶斯训练评估 + 与 LLM 对比
│  └─ predict.py          # 预测封装
├─ requirements.txt
├─ .env                   # DeepSeek API Key（已被 .gitignore 排除，不提交）
└─ README.md
```

## 数据
公开中文情感数据集 **online_shopping_10_cats**（电商商品评论，覆盖书籍/平板/手机/水果/洗发水等 10 类目）：共 **62724** 条（清洗去重后），标签 `1`=正向 31707 条、`0`=负向 31017 条，正负均衡。

## 快速开始
```bash
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
source .venv/bin/pip install -r requirements.txt          # macOS/Linux

# 2. 配置 DeepSeek Key（复制 .env 示例）
#    API_KEY=sk-xxx  BASE_URL=https://api.deepseek.com  MODEL=deepseek-chat

# 3. 跑完整流程：大模型标注训练集 → 训练朴素贝叶斯 → 与大模型对比
.venv\Scripts\python src\train.py

# 4. 预测
.venv\Scripts\python src\predict.py "这家酒店服务很好，房间干净舒适。"
#   -> 正向
```

## 结果（一次真实运行）
| 指标 | 数值 |
|---|---|
| 朴素贝叶斯 训练集 5 折交叉验证 | **0.7733** ± 0.0512 |
| 朴素贝叶斯 评估集准确率 | **0.7850** |
| DeepSeek 大模型 评估集准确率 | **0.9550** |

结论：在 6.3 万条电商评论上，DeepSeek 大模型（0.955）判别显著优于朴素贝叶斯（0.785），验证了大模型在中文情感判别上的强泛化能力；训练集由大模型批量弱监督标注（300 条），无需人工标注即可构建分类器。

## 说明
- `.env` 含 API Key，已加入 `.gitignore`，不会被提交。
- `model.joblib` / `vectorizer.joblib` 为训练产物，可直接用于预测。
- 标注数量可通过 `src/train.py` 的 `n_train` / `n_eval` 调整，支持增量扩充后重训。
