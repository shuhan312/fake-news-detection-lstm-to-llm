"""
结果分析工具（notebook 02、03 共用）/ Result-analysis helpers shared by notebooks 02 and 03.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments import load_probs
from src.external import load_welfake_raw


def publisher_groups(welfake_test: pd.DataFrame) -> pd.Series:
    """给 WELFake test 的每篇文章分组：假新闻一组，真新闻按出版方分组。
    Group each WELFake test article: fake articles together, real ones by publisher.

    出版方后缀在清洗时已经删掉了，所以这里回到 *原始* 标题去识别。
    Publisher suffixes were stripped during cleaning, so the RAW titles are used.

    Returns:
        以 doc_id 为索引的 Series，值为 'FAKE articles (all)' / 'REAL: The New York Times' /
        'REAL: Breitbart' / 'REAL: other outlets'。
    """
    raw_titles = load_welfake_raw().set_index('doc_id').loc[welfake_test['doc_id'], 'title']
    publisher = raw_titles.str.extract(r' - (The New York Times|Breitbart)\s*$')[0] \
                          .fillna('other outlets').values
    group = np.where(welfake_test['label'] == 0, 'FAKE articles (all)', 'REAL: ' + publisher)
    return pd.Series(group, index=welfake_test['doc_id'].values)


def predicted_real_by_group(runs: dict, groups: pd.Series) -> pd.DataFrame:
    """每个模型在每个组里判为"真"的比例（%），多个种子取平均。
    Share predicted real (%) per group for each model, averaged over seeds.

    Args:
        runs: {显示名称: (model, config, [seeds])}
        groups: publisher_groups() 的结果 / output of publisher_groups()
    """
    table = {}
    for label, (model, config, seeds) in runs.items():
        shares = []
        for seed in seeds:
            ids, prob = load_probs(model, config, seed)['WELFake test']
            shares.append(pd.Series(prob > 0.5, index=ids).groupby(groups.loc[ids].values).mean() * 100)
        table[label] = pd.concat(shares, axis=1).mean(axis=1).round(1)
    out = pd.DataFrame(table)
    # 文章数作为一 *列* 加入（按组名对齐）；加成一行会按模型名对齐，结果全是 NaN
    # Article counts go in as a COLUMN (aligned on group names); as a row they
    # would align on model names and come out as NaN
    out.insert(0, 'n articles', groups.value_counts())
    return out


def aligned_probs(run_a, run_b, eval_set: str):
    """取出两次运行在同一评估集上的概率，并按 doc_id 对齐（防止顺序不一致导致错配）。
    Probabilities of two runs on one eval set, aligned by doc_id."""
    ids_a, p_a = load_probs(*run_a)[eval_set]
    ids_b, p_b = load_probs(*run_b)[eval_set]
    b = pd.Series(p_b, index=ids_b).loc[ids_a].to_numpy()
    return ids_a, p_a, b


def paired_bootstrap_auc_diff(labels, prob_a, prob_b, n_boot: int = 1000, seed: int = 42) -> dict:
    """配对 bootstrap：两个模型在 *同一批* 重抽样文章上的 AUC 差（A − B）及 95% 置信区间。
    Paired bootstrap of the AUC difference (A − B) on the SAME resampled articles.

    为什么要"配对"：两个模型在同一批文章上评估，难的文章对两个模型都难。每次重抽样时用同一批文章
    计算两个 AUC 再相减，可以去掉"这批文章难不难"带来的共同波动，比分别算置信区间更灵敏。
    Why paired: hard articles are hard for both models; differencing on the same
    resample removes that shared noise, so the test is more sensitive.

    置信区间不包含 0，说明差距在统计上可靠；包含 0，说明分不出高下。
    A CI excluding 0 means a reliable difference; including 0 means no detectable one.
    """
    rng = np.random.default_rng(seed)
    labels, prob_a, prob_b = map(np.asarray, (labels, prob_a, prob_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        diffs.append(roc_auc_score(labels[idx], prob_a[idx]) - roc_auc_score(labels[idx], prob_b[idx]))
    return {'AUC difference (A - B)': round(roc_auc_score(labels, prob_a) - roc_auc_score(labels, prob_b), 3),
            '95% CI': (round(float(np.percentile(diffs, 2.5)), 3),
                       round(float(np.percentile(diffs, 97.5)), 3))}
