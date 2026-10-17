"""
错误分析工具（notebook 09）/ Error-analysis helpers for notebook 09.

前面的 notebook 回答"分数是多少"，这里回答"错在哪里、为什么错"。所有分析都在 LIAR test 上进行，
只是 *描述* 已经固定的模型，不再调整任何东西，所以不存在"在测试集上调参"的问题。
Earlier notebooks ask "how well"; this one asks "where and why it fails". The
analysis describes models that are already fixed on LIAR test; nothing is tuned.

阈值 / Thresholds
    比较两个模型"谁判对了"时，阈值必须公平：Haiku 原始概率偏向"假"（只把 42% 判为真），
    直接用 0.5 会让它显得更差。所以每个单独的模型都先在 val 上校准（和 5c 的单模型候选完全
    相同：一个特征的逻辑回归），再用 0.5 作为分界线。
    Comparing "who got it right" needs fair thresholds: Haiku's raw probabilities
    lean towards "false", so every single model is first calibrated on val (a
    one-feature logistic regression, exactly the single-model candidates of 5c).
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ensemble import load_scores, make_combiner, to_logit

# 含数字或数量的说法（"失业率翻了一倍"、"3 million"）：需要具体数据才能核查
# Numbers or quantities: checking them needs specific data
_RE_NUMBER = re.compile(
    r'\d|\b(?:percent|million|billion|trillion|thousand|hundred|half|twice|double[sd]?|triple[sd]?|'
    r'quadrupled?|one-third|one-fourth|one-quarter|majority)\b', re.I)

# 绝对化的说法（"从来没有"、"每一个"、"第一个"）：只要有一个反例就不成立
# Absolute claims ("never", "every", "first"): one counter-example makes them false
_RE_ABSOLUTE = re.compile(
    r'\b(?:never|always|every|everyone|nobody|no one|none|only|first|last|largest|highest|lowest|'
    r'most|least|biggest|worst|best|record|ever|all)\b', re.I)

# 依赖上下文的指代（"他"、"这项法案"、"我的对手"）：不知道指的是谁 / 什么，就很难判断
# Context-dependent references: without knowing who or what is meant, the claim is hard to judge
_RE_CONTEXT = re.compile(
    r'\b(?:he|she|his|her|him|they|their|them|my opponent|this (?:bill|law|plan|proposal|budget|amendment|'
    r'measure|legislation)|the (?:bill|plan|proposal|governor|president|senator|mayor|congressman))\b', re.I)


def statement_features(text: pd.Series) -> pd.DataFrame:
    """每条陈述的几个简单特征，用来给错误分组。都是粗略的规则，只用于描述，不作为模型输入。
    A few simple per-statement features for grouping errors. Rough rules,
    descriptive only, never model inputs."""
    words = text.str.split().str.len()
    return pd.DataFrame({
        'length': pd.cut(words, [0, 12, 20, 30, 1000], labels=['1-12 words', '13-20', '21-30', '31+']),
        'has number / quantity': text.str.contains(_RE_NUMBER),
        'absolute claim': text.str.contains(_RE_ABSOLUTE),
        'context-dependent reference': text.str.contains(_RE_CONTEXT),
        # PolitiFact 用 "Says ..." 转述别人的说法（常见于广告、社交媒体）
        # PolitiFact's "Says ..." paraphrases (ads, social media posts, chain emails)
        'starts with "Says"': text.str.match(r'^\s*Says\b'),
    }, index=text.index)


def calibrated(name: str) -> pd.Series:
    """单个模型在 test 上的概率，先在 val 上校准（一个特征的逻辑回归）。
    A single model's test probabilities after calibration on val."""
    val, test = load_scores('val', names=[name]), load_scores('test', names=[name])
    model = make_combiner().fit(to_logit(val[[name]].to_numpy()), val['label'])
    return pd.Series(model.predict_proba(to_logit(test[[name]].to_numpy()))[:, 1], index=test.index)


def group_table(y: pd.Series, preds: dict, groups: pd.Series, min_n: int = 30) -> pd.DataFrame:
    """按组比较各模型：数量、真陈述比例、各模型的准确率和 AUC。少于 min_n 条的组不显示（太少，不稳定）。
    Per-group n, real share, and each model's accuracy and AUC; groups smaller
    than min_n are dropped as too noisy."""
    rows = {}
    for g, idx in groups.groupby(groups, observed=True).groups.items():
        if len(idx) < min_n:
            continue
        yy = y.loc[idx]
        row = {'n': len(idx), 'real %': round(100 * yy.mean(), 1)}
        for name, p in preds.items():
            pp = p.loc[idx]
            row[f'{name}: accuracy %'] = round(100 * ((pp > 0.5) == yy).mean(), 1)
            row[f'{name}: AUC'] = round(roc_auc_score(yy, pp), 3) if yy.nunique() == 2 else np.nan
        rows[g] = row
    return pd.DataFrame(rows).T


def agreement_quadrants(y: pd.Series, a: pd.Series, b: pd.Series, name_a: str, name_b: str) -> pd.Series:
    """两个模型（都已校准）在每条陈述上谁判对了：都对 / 只有 A 对 / 只有 B 对 / 都错。
    Which of two calibrated models is right on each statement."""
    right_a, right_b = (a > 0.5) == y, (b > 0.5) == y
    return pd.Series(np.select([right_a & right_b, right_a & ~right_b, ~right_a & right_b],
                               ['both right', f'only {name_a} right', f'only {name_b} right'], 'both wrong'),
                     index=y.index)


def bias_by_group(y: pd.Series, preds: dict, groups: pd.Series, min_n: int = 30) -> pd.DataFrame:
    """每组"判为真的比例"减去"实际真的比例"。正数 = 对这一组偏宽松，负数 = 偏严格。
    Predicted-real share minus actual real share per group: positive = too
    lenient towards the group, negative = too harsh."""
    rows = {}
    for g, idx in groups.groupby(groups, observed=True).groups.items():
        if len(idx) < min_n:
            continue
        actual = 100 * y.loc[idx].mean()
        rows[g] = {'n': len(idx), 'actual real %': round(actual, 1),
                   **{f'{name}: predicted - actual': round(100 * (p.loc[idx] > 0.5).mean() - actual, 1)
                      for name, p in preds.items()}}
    return pd.DataFrame(rows).T
