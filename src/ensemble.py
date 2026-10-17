"""
第 5c 步：把只看文字的模型和 LLM 组合起来 / Step 5c: combining the text models with the LLM.

为什么组合 / Why combine
    在 LIAR test 上，Haiku（5a）和 DistilBERT 的 AUC 差不多（0.655 对 0.68），但两者的打分
    几乎不相关（Spearman 0.21；TF-IDF 和 DistilBERT 之间是 0.77）。DistilBERT 学的是
    "说法的样子"，Haiku 用的是世界知识。两个水平接近、依据不同的模型，组合后通常比任何一个都好。
    这个相关系数只用了预测值，没有用测试集标签。
    Haiku (5a) and DistilBERT have similar test AUCs but nearly uncorrelated
    scores (Spearman 0.21, vs 0.77 between TF-IDF and DistilBERT): one reads
    phrasing, the other uses world knowledge. Such models usually combine well.
    The correlation uses predictions only, not test labels.

怎么组合 / How
    把每个模型的概率转成 logit，用逻辑回归学习每个模型的权重（"stacking"）。
    组合器只在 LIAR val 上训练：DistilBERT 和 TF-IDF 已经见过训练集，它们在训练集上的打分
    过于自信，拿来学权重会高估它们。
    Each model's probability becomes a logit; a logistic regression learns the
    weights ("stacking"). The combiner is fitted on LIAR val only: DistilBERT and
    TF-IDF have seen the training set, where their scores are overconfident.

    逻辑回归同时会重新校准阈值：5b 的问题之一是模型整体偏向判假（只把 21% 判为真），
    学到的截距会把分界线移回合适的位置，所以准确率也会随之修正。
    The regression also recalibrates the threshold (its intercept), which fixes
    accuracy when a model is shifted towards "false", as 5b was.

选择规则（事先定好，只用 val）/ Selection rule (fixed in advance, val only)
    每个候选组合在 val 上做 5 折交叉验证，计算"组合器没见过的那一折"上的 AUC。
    不能在 val 上训练再在 val 上打分：那样分数会偏高，而且参数多的组合会占便宜。
    DistilBERT 有 3 个种子，每个种子分别组合；选 3 个种子平均交叉验证 AUC 最高的候选。
    测试集只用来评估这个被选中的候选，评估之后不再修改。
    Every candidate gets a 5-fold cross-validated AUC on val (scored on the fold
    the combiner did not see; fitting and scoring on the same val rows would
    flatter combinations with more parameters). DistilBERT has 3 seeds, each
    combined separately; the candidate with the best mean CV AUC is chosen. The
    test set only evaluates that one candidate, and nothing is changed afterwards.

已知的偏差 / Known bias
    DistilBERT 的最佳 epoch、TF-IDF 的 C 都是在 val 上选的，所以它们在 val 上的分数略微偏高，
    组合器给它们的权重可能也略微偏大。这只会让组合 *低估* LLM 的作用，不会让结果虚高。
    DistilBERT's epoch and TF-IDF's C were chosen on val, so their val scores are
    slightly flattering and may get slightly too much weight. This can only make
    the combination understate the LLM, not inflate the result.

命令行 / Command line:
    python -m src.ensemble --select    # 只用 val：比较候选并保存选择 / val only: compare and save the choice
    python -m src.ensemble --test      # 最终评估：只评估被选中的候选 / final: evaluate the chosen candidate
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import baselines as B
from src.analysis import paired_bootstrap_auc_diff
from src.data import PROJECT_ROOT
from src.external import load_liar

LIAR_DIR = PROJECT_ROOT / 'results' / 'liar'
LLM_DIR = PROJECT_ROOT / 'results' / 'llm'
SEEDS = (42, 43, 44)

# 每个模型的预测文件：{名字: {划分: (npz 文件, 键名)}}
# DistilBERT 和 TF-IDF 在一个文件里同时保存了 val 和 test；LLM 的 val 和 test 是两个文件
# Prediction files per model. DistilBERT / TF-IDF keep val and test in one file;
# the LLM has one file per split (see run_name in src/llm_factcheck.py)
SOURCES = {
    'tfidf': {s: (LIAR_DIR / 'tfidf_liar.npz', f'LIAR {s}') for s in ('val', 'test')},
    **{f'distilbert_seed{seed}': {s: (LIAR_DIR / f'distilbert_seed{seed}.npz', f'LIAR {s}')
                                  for s in ('val', 'test')} for seed in SEEDS},
    'haiku': {'val': (LLM_DIR / 'llm_no_evidence_val.npz', 'LIAR val'),
              'test': (LLM_DIR / 'llm_no_evidence.npz', 'LIAR test')},
}

# 候选组合；'distilbert' 会替换成每个种子各自的列
# Candidates; 'distilbert' is replaced by each seed's own column
CANDIDATES = {
    'TF-IDF': ['tfidf'],
    'DistilBERT': ['distilbert'],
    'Haiku': ['haiku'],
    'DistilBERT + TF-IDF': ['distilbert', 'tfidf'],
    'TF-IDF + Haiku': ['tfidf', 'haiku'],
    'DistilBERT + Haiku': ['distilbert', 'haiku'],
    'DistilBERT + TF-IDF + Haiku': ['distilbert', 'tfidf', 'haiku'],
}

N_FOLDS = 5
SELECTION_PATH = LLM_DIR / 'combination_val.json'
TEST_PATH = LLM_DIR / 'combination_test.json'


def load_scores(split: str, sources: dict = None) -> pd.DataFrame:
    """读出所有模型在一个划分上的概率，按 doc_id 对齐，并加上标签。
    All models' probabilities on one split, aligned by doc_id, plus the label.

    各文件里的陈述顺序不一定相同，所以一律按 doc_id 对齐；任何一个模型缺了某条陈述都会报错，
    而不是悄悄地按位置错配。
    Files may store statements in different orders, so everything is aligned by
    doc_id; a statement missing from any model raises instead of mis-pairing.
    """
    sources = sources or SOURCES
    liar = load_liar()
    rows = liar[liar['split'] == split].set_index('doc_id')
    out = pd.DataFrame({'label': rows['label']})
    for name, files in sources.items():
        path, key = files[split]
        z = np.load(path, allow_pickle=True)
        scores = pd.Series(z[f'prob::{key}'], index=z[f'doc_id::{key}'].astype(str))
        missing = out.index.difference(scores.index)
        if len(missing):
            raise ValueError(f'{name} has no prediction for {len(missing)} {split} statements')
        out[name] = scores.loc[out.index].to_numpy()
    return out


def to_logit(prob, eps: float = 0.01) -> np.ndarray:
    """概率 → logit。先截断到 [0.01, 0.99]：Haiku 会回答 0 或 100，直接取 logit 会得到无穷大。
    Probability → logit, clipped to [0.01, 0.99] first: Haiku can answer 0 or 100,
    whose logit is infinite."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def make_combiner():
    """组合器：标准化后的 logit → 逻辑回归。C=1 是适度的正则化，最多只有 3 个特征，结果对 C 不敏感。
    Combiner: standardised logits → logistic regression. C=1 is mild
    regularisation; with at most 3 features the result is insensitive to C."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))


def columns_for(candidate: str, seed: int) -> list:
    """候选组合在某个 DistilBERT 种子下用到的列 / Columns of a candidate for one DistilBERT seed."""
    return [f'distilbert_seed{seed}' if c == 'distilbert' else c for c in CANDIDATES[candidate]]


def cv_auc(scores: pd.DataFrame, columns: list, n_folds: int = N_FOLDS, seed: int = 42) -> float:
    """在 val 上做分层 K 折交叉验证：每一折的预测都来自没见过这一折的组合器，再计算整体 AUC。
    Stratified K-fold on val: every prediction comes from a combiner that did not
    see that fold; the AUC is computed over all out-of-fold predictions."""
    X, y = to_logit(scores[columns].to_numpy()), scores['label'].to_numpy()
    oof = np.zeros(len(y))
    for train_idx, held_idx in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        model = make_combiner().fit(X[train_idx], y[train_idx])
        oof[held_idx] = model.predict_proba(X[held_idx])[:, 1]
    return roc_auc_score(y, oof)


def select(val: pd.DataFrame = None) -> dict:
    """只用 val 比较所有候选，按事先定好的规则选出一个，并保存。
    Compare every candidate on val only, choose one by the fixed rule, save it."""
    val = load_scores('val') if val is None else val
    table = {}
    for candidate in CANDIDATES:
        per_seed = {seed: cv_auc(val, columns_for(candidate, seed)) for seed in SEEDS}
        table[candidate] = {**{f'seed {s}': round(a, 4) for s, a in per_seed.items()},
                            'mean CV AUC': round(float(np.mean(list(per_seed.values()))), 4)}
    chosen = max(table, key=lambda c: table[c]['mean CV AUC'])
    # 和 Haiku 的相关性也记录下来（只是描述，不参与选择）/ correlations, descriptive only
    corr = val.drop(columns='label').corr(method='spearman').round(2)
    record = {'split': 'val', 'n': len(val), 'real %': round(100 * val['label'].mean(), 1),
              'rule': 'highest mean 5-fold CV AUC on LIAR val over DistilBERT seeds',
              'cv_auc': table, 'chosen': chosen, 'spearman': corr.to_dict()}
    SELECTION_PATH.write_text(json.dumps(record, indent=2))
    return record


def evaluate_test(n_boot: int = 1000) -> dict:
    """最终评估：只评估 val 上选中的候选。组合器在全部 val 上重新训练，然后在 test 上打分；
    每个 DistilBERT 种子分别和"这个种子的 DistilBERT 单独使用"做配对比较。
    Final evaluation of the candidate chosen on val only: refit the combiner on
    all of val, score test, and compare with the same seed's DistilBERT alone.
    """
    if not SELECTION_PATH.exists():
        raise FileNotFoundError('run --select first: the candidate must be chosen on val')
    chosen = json.loads(SELECTION_PATH.read_text())['chosen']
    val, test = load_scores('val'), load_scores('test')
    y = test['label'].to_numpy()
    per_seed = {}
    for seed in SEEDS:
        cols = columns_for(chosen, seed)
        model = make_combiner().fit(to_logit(val[cols].to_numpy()), val['label'])
        prob = model.predict_proba(to_logit(test[cols].to_numpy()))[:, 1]
        db = test[f'distilbert_seed{seed}'].to_numpy()
        weights = model[-1].coef_[0]
        per_seed[f'seed {seed}'] = {
            'columns': cols,
            'weights (standardised logits)': dict(zip(cols, np.round(weights, 3).tolist())),
            'combined': B.evaluate(y, prob),
            'DistilBERT alone': B.evaluate(y, db),
            'combined - DistilBERT': paired_bootstrap_auc_diff(y, prob, db, n_boot=n_boot),
            'combined - TF-IDF': paired_bootstrap_auc_diff(y, prob, test['tfidf'].to_numpy(), n_boot=n_boot),
            'combined - Haiku': paired_bootstrap_auc_diff(y, prob, test['haiku'].to_numpy(), n_boot=n_boot),
        }
        per_seed[f'seed {seed}']['prob'] = prob.round(4).tolist()
    record = {'split': 'test', 'chosen_on_val': chosen, 'doc_ids': test.index.tolist(), 'results': per_seed}
    TEST_PATH.write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description='Step 5c: combine text models with the LLM.')
    parser.add_argument('--select', action='store_true', help='compare candidates on LIAR val')
    parser.add_argument('--test', action='store_true', help='evaluate the chosen candidate on LIAR test')
    args = parser.parse_args()
    if args.select:
        record = select()
        print(pd.DataFrame(record['cv_auc']).T.sort_values('mean CV AUC', ascending=False))
        print('chosen on val:', record['chosen'])
    if args.test:
        record = evaluate_test()
        for seed, r in record['results'].items():
            print(seed, r['weights (standardised logits)'], '| combined', r['combined']['AUC'],
                  r['combined']['accuracy %'], '| DistilBERT', r['DistilBERT alone']['AUC'],
                  '| diff', r['combined - DistilBERT'])


if __name__ == '__main__':
    main()
