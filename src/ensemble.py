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
    python -m src.ensemble --select5d  # 5d：Sonnet 能否让组合进一步提高（只用 val）/ 5d on val
    python -m src.ensemble --test5d    # 5d：只有 val 上提升 >= 0.01 才允许运行 / only if the val gain >= 0.01
    python -m src.ensemble --speaker-prior                 # 5f 的捷径基线 / the 5f shortcut baseline
    python -m src.ensemble --select-ext 5f --test-ext 5f   # 5f（5e 同理）/ 5f (likewise 5e)
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
    # 5d：更大的模型，提示词和 Haiku 5a 完全相同 / 5d: the larger model, same prompt as Haiku 5a
    'sonnet': {'val': (LLM_DIR / 'llm_no_evidence_sonnet_val.npz', 'LIAR val'),
               'test': (LLM_DIR / 'llm_no_evidence_sonnet.npz', 'LIAR test')},
    # 5f：Haiku + 发言背景 / Haiku given the speaker and setting
    'haiku_context': {'val': (LLM_DIR / 'llm_context_val.npz', 'LIAR val'),
                      'test': (LLM_DIR / 'llm_context.npz', 'LIAR test')},
    # 5e：Haiku + 维基百科证据（5b）/ Haiku with Wikipedia evidence (5b)
    'haiku_evidence': {'val': (LLM_DIR / 'llm_evidence_val.npz', 'LIAR val'),
                       'test': (LLM_DIR / 'llm_evidence.npz', 'LIAR test')},
    # 诊断用的捷径基线，*不* 作为可选模型：说话人在训练集里说真话的比例（build_speaker_prior）
    # Diagnostic shortcut baseline, NOT a selectable model: the speaker's true rate in train
    'speaker_prior': {s: (LLM_DIR / 'speaker_prior.npz', f'LIAR {s}') for s in ('val', 'test')},
}

# 5c 的候选组合；'distilbert' 会替换成每个种子各自的列
# 5c candidates; 'distilbert' is replaced by each seed's own column
CANDIDATES = {
    'TF-IDF': ['tfidf'],
    'DistilBERT': ['distilbert'],
    'Haiku': ['haiku'],
    'DistilBERT + TF-IDF': ['distilbert', 'tfidf'],
    'TF-IDF + Haiku': ['tfidf', 'haiku'],
    'DistilBERT + Haiku': ['distilbert', 'haiku'],
    'DistilBERT + TF-IDF + Haiku': ['distilbert', 'tfidf', 'haiku'],
}

# 5d 的候选：5c 选中的组合作为参照，再加上用 Sonnet 替换 Haiku、或者两个都用
# 5d candidates: the 5c choice as the reference, Sonnet instead of Haiku, or both
REFERENCE_5D = 'DistilBERT + TF-IDF + Haiku'
CANDIDATES_5D = {
    'Haiku': ['haiku'],
    'Sonnet': ['sonnet'],
    REFERENCE_5D: ['distilbert', 'tfidf', 'haiku'],
    'DistilBERT + TF-IDF + Sonnet': ['distilbert', 'tfidf', 'sonnet'],
    'DistilBERT + TF-IDF + Haiku + Sonnet': ['distilbert', 'tfidf', 'haiku', 'sonnet'],
}

# 5d 的事先规则：含 Sonnet 的最佳组合在 val 上的平均交叉验证 AUC 必须比参照高至少 0.01，
# 才值得花钱在 test 上运行 Sonnet 并评估一次；否则结论就是"更大的模型没有带来明显提升"。
# 0.01 大约是 val 上种子之间的波动（5c 中 3 个种子相差 0.008），低于它的差距很可能只是噪声。
# 5d rule, fixed before the results: the best Sonnet candidate must beat the
# reference by >= 0.01 mean CV AUC on val before Sonnet is run and evaluated on
# test; 0.01 is about the seed-to-seed spread on val (0.008 in 5c).
MIN_GAIN_5D = 0.01

N_FOLDS = 5
SELECTION_PATH = LLM_DIR / 'combination_val.json'
TEST_PATH = LLM_DIR / 'combination_test.json'
SELECTION_5D_PATH = LLM_DIR / 'combination_sonnet_val.json'
TEST_5D_PATH = LLM_DIR / 'combination_sonnet_test.json'

# 5f 和 5e 用同一套规则（和 5d 相同的 MIN_GAIN_5D = 0.01），各回答两个问题，都只在 val 上判断：
#   Q1：能否让 5c 的组合（只用 Haiku）提高？
#   Q2：在 5d 的 Sonnet 组合（目前最好的）之上，还能否增加信息？
# 只有达到 0.01 的问题，才在 test 上评估一次。不在 questions 里的候选只用来解释，永远不会被选中。
# 5f and 5e share the 5d rule and each ask two questions on val: Q1 does it
# improve the 5c combination, Q2 does it add anything on top of the 5d Sonnet
# combination. Only questions reaching 0.01 are evaluated on test; candidates
# outside the questions are diagnostic and never selectable.
BEST_5D = 'DistilBERT + TF-IDF + Haiku + Sonnet'
EXTENSIONS = {
    '5f': {
        'file': 'context',
        'candidates': {
            'Haiku': ['haiku'],
            'Haiku + context': ['haiku_context'],
            'speaker prior (shortcut)': ['speaker_prior'],
            REFERENCE_5D: ['distilbert', 'tfidf', 'haiku'],
            'DistilBERT + TF-IDF + Haiku context': ['distilbert', 'tfidf', 'haiku_context'],
            'DistilBERT + TF-IDF + Haiku + Haiku context': ['distilbert', 'tfidf', 'haiku', 'haiku_context'],
            'DistilBERT + TF-IDF + Haiku + speaker prior (shortcut)': ['distilbert', 'tfidf', 'haiku', 'speaker_prior'],
            BEST_5D: ['distilbert', 'tfidf', 'haiku', 'sonnet'],
            BEST_5D + ' + Haiku context': ['distilbert', 'tfidf', 'haiku', 'sonnet', 'haiku_context'],
            BEST_5D + ' + speaker prior (shortcut)': ['distilbert', 'tfidf', 'haiku', 'sonnet', 'speaker_prior'],
        },
        'questions': {
            'Q1: improves the 5c combination': (REFERENCE_5D, ['DistilBERT + TF-IDF + Haiku context',
                                                              'DistilBERT + TF-IDF + Haiku + Haiku context']),
            'Q2: adds to the 5d Sonnet combination': (BEST_5D, [BEST_5D + ' + Haiku context']),
        },
    },
    '5e': {
        'file': 'evidence',
        'candidates': {
            'Haiku': ['haiku'],
            'Haiku + evidence': ['haiku_evidence'],
            REFERENCE_5D: ['distilbert', 'tfidf', 'haiku'],
            'DistilBERT + TF-IDF + Haiku + Haiku evidence': ['distilbert', 'tfidf', 'haiku', 'haiku_evidence'],
            BEST_5D: ['distilbert', 'tfidf', 'haiku', 'sonnet'],
            BEST_5D + ' + Haiku evidence': ['distilbert', 'tfidf', 'haiku', 'sonnet', 'haiku_evidence'],
        },
        'questions': {
            'Q1: improves the 5c combination': (REFERENCE_5D, ['DistilBERT + TF-IDF + Haiku + Haiku evidence']),
            'Q2: adds to the 5d Sonnet combination': (BEST_5D, [BEST_5D + ' + Haiku evidence']),
        },
    },
}


def load_scores(split: str, sources: dict = None, names=None) -> pd.DataFrame:
    """读出模型在一个划分上的概率，按 doc_id 对齐，并加上标签。names 指定只读哪些模型。
    Models' probabilities on one split, aligned by doc_id, plus the label;
    `names` restricts which models are read.

    各文件里的陈述顺序不一定相同，所以一律按 doc_id 对齐；任何一个模型缺了某条陈述都会报错，
    而不是悄悄地按位置错配。
    Files may store statements in different orders, so everything is aligned by
    doc_id; a statement missing from any model raises instead of mis-pairing.
    """
    sources = sources or SOURCES
    names = list(sources) if names is None else list(names)
    liar = load_liar()
    rows = liar[liar['split'] == split].set_index('doc_id')
    out = pd.DataFrame({'label': rows['label']})
    for name in names:
        path, key = sources[name][split]
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
    """组合器：标准化后的 logit → 逻辑回归。C=1 是适度的正则化，最多只有 4 个特征，结果对 C 不敏感。
    Combiner: standardised logits → logistic regression. C=1 is mild
    regularisation; with at most 4 features the result is insensitive to C."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))


def columns_for(candidate: str, seed: int, candidates: dict = None) -> list:
    """候选组合在某个 DistilBERT 种子下用到的列 / Columns of a candidate for one DistilBERT seed."""
    candidates = candidates or CANDIDATES
    return [f'distilbert_seed{seed}' if c == 'distilbert' else c for c in candidates[candidate]]


def _models_needed(candidates: dict) -> list:
    """候选组合用到的所有模型（DistilBERT 展开成 3 个种子）/ All models the candidates use."""
    names = []
    for cols in candidates.values():
        for c in cols:
            for n in ([f'distilbert_seed{s}' for s in SEEDS] if c == 'distilbert' else [c]):
                if n not in names:
                    names.append(n)
    return names


def cv_auc(scores: pd.DataFrame, columns: list, n_folds: int = N_FOLDS, seed: int = 42) -> float:
    """在 val 上做分层 K 折交叉验证：每一折的预测都来自没见过这一折的组合器，再计算整体 AUC。
    折的划分固定（seed=42），所以不同候选、不同步骤（5c、5d）都在完全相同的折上比较。
    Stratified K-fold on val: every prediction comes from a combiner that did not
    see that fold. The folds are fixed (seed 42), so 5c and 5d candidates are
    compared on exactly the same folds."""
    X, y = to_logit(scores[columns].to_numpy()), scores['label'].to_numpy()
    oof = np.zeros(len(y))
    for train_idx, held_idx in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        model = make_combiner().fit(X[train_idx], y[train_idx])
        oof[held_idx] = model.predict_proba(X[held_idx])[:, 1]
    return roc_auc_score(y, oof)


def compare_on_val(val: pd.DataFrame, candidates: dict) -> dict:
    """每个候选在每个 DistilBERT 种子下的交叉验证 AUC，以及 3 个种子的平均值。
    CV AUC of every candidate for each DistilBERT seed, plus the mean over seeds."""
    table = {}
    for candidate in candidates:
        per_seed = {seed: cv_auc(val, columns_for(candidate, seed, candidates)) for seed in SEEDS}
        table[candidate] = {**{f'seed {s}': round(a, 4) for s, a in per_seed.items()},
                            'mean CV AUC': round(float(np.mean(list(per_seed.values()))), 4)}
    return table


def select(val: pd.DataFrame = None) -> dict:
    """5c：只用 val 比较所有候选，按事先定好的规则选出一个，并保存。
    5c: compare every candidate on val only, choose one by the fixed rule, save it."""
    val = load_scores('val', names=_models_needed(CANDIDATES)) if val is None else val
    table = compare_on_val(val, CANDIDATES)
    chosen = max(table, key=lambda c: table[c]['mean CV AUC'])
    # 和 Haiku 的相关性也记录下来（只是描述，不参与选择）/ correlations, descriptive only
    corr = val.drop(columns='label').corr(method='spearman').round(2)
    record = {'split': 'val', 'n': len(val), 'real %': round(100 * val['label'].mean(), 1),
              'rule': 'highest mean 5-fold CV AUC on LIAR val over DistilBERT seeds',
              'cv_auc': table, 'chosen': chosen, 'spearman': corr.to_dict()}
    SELECTION_PATH.write_text(json.dumps(record, indent=2))
    return record


def select_5d(val: pd.DataFrame = None) -> dict:
    """5d：只用 val 判断 Sonnet 能否让组合进一步提高，并按 MIN_GAIN_5D 决定是否值得在 test 上运行。
    5d: on val only, does Sonnet improve the combination, and is the gain large
    enough (MIN_GAIN_5D) to justify running Sonnet on test?"""
    val = load_scores('val', names=_models_needed(CANDIDATES_5D)) if val is None else val
    table = compare_on_val(val, CANDIDATES_5D)
    with_sonnet = [c for c, cols in CANDIDATES_5D.items() if 'sonnet' in cols and len(cols) > 1]
    best = max(with_sonnet, key=lambda c: table[c]['mean CV AUC'])
    gain = round(table[best]['mean CV AUC'] - table[REFERENCE_5D]['mean CV AUC'], 4)
    corr = val.drop(columns='label').corr(method='spearman').round(2)
    record = {'split': 'val', 'n': len(val), 'reference': REFERENCE_5D,
              'rule': f'run Sonnet on test only if the best Sonnet combination beats the reference '
                      f'by at least {MIN_GAIN_5D} mean 5-fold CV AUC on LIAR val',
              'cv_auc': table, 'best_with_sonnet': best, 'gain_over_reference': gain,
              'run_test': bool(gain >= MIN_GAIN_5D), 'chosen': best if gain >= MIN_GAIN_5D else REFERENCE_5D,
              'single_model_val_auc': {m: round(roc_auc_score(val['label'], val[m]), 4)
                                       for m in ('haiku', 'sonnet')},
              'spearman': corr.to_dict()}
    SELECTION_5D_PATH.write_text(json.dumps(record, indent=2))
    return record


def _evaluate_chosen(chosen: str, candidates: dict, n_boot: int, reference: dict = None) -> dict:
    """组合器在全部 val 上重新训练，在 test 上打分，并和单个模型做配对比较。
    reference：{种子: 参照组合在 test 上的概率}，用于 5d 和 5c 的组合直接比较。
    Refit on all of val, score test, paired comparisons with single models;
    `reference` holds a reference combination's test probabilities per seed."""
    names = _models_needed({chosen: candidates[chosen]}) + ['tfidf', 'haiku']
    names = list(dict.fromkeys(names + [f'distilbert_seed{s}' for s in SEEDS]))
    val, test = load_scores('val', names=names), load_scores('test', names=names)
    y = test['label'].to_numpy()
    per_seed = {}
    for seed in SEEDS:
        cols = columns_for(chosen, seed, candidates)
        model = make_combiner().fit(to_logit(val[cols].to_numpy()), val['label'])   # 只用 val 训练 / fit on val
        prob = model.predict_proba(to_logit(test[cols].to_numpy()))[:, 1]          # 在 test 上打分 / score test
        db = test[f'distilbert_seed{seed}'].to_numpy()
        weights = model[-1].coef_[0]
        r = {'columns': cols,
             'weights (standardised logits)': dict(zip(cols, np.round(weights, 3).tolist())),
             'combined': B.evaluate(y, prob),
             'DistilBERT alone': B.evaluate(y, db),
             'combined - DistilBERT': paired_bootstrap_auc_diff(y, prob, db, n_boot=n_boot),
             'combined - TF-IDF': paired_bootstrap_auc_diff(y, prob, test['tfidf'].to_numpy(), n_boot=n_boot),
             'combined - Haiku': paired_bootstrap_auc_diff(y, prob, test['haiku'].to_numpy(), n_boot=n_boot)}
        if reference is not None:
            r['combined - reference'] = paired_bootstrap_auc_diff(y, prob, np.asarray(reference[seed]), n_boot=n_boot)
        r['prob'] = prob.round(4).tolist()
        per_seed[f'seed {seed}'] = r
    return {'doc_ids': test.index.tolist(), 'results': per_seed}


def evaluate_test(n_boot: int = 1000) -> dict:
    """5c 的最终评估：只评估 val 上选中的候选；每个 DistilBERT 种子分别和"这个种子的 DistilBERT 单独使用"比较。
    5c final evaluation of the candidate chosen on val only, compared with the
    same seed's DistilBERT alone.
    """
    if not SELECTION_PATH.exists():
        raise FileNotFoundError('run --select first: the candidate must be chosen on val')
    chosen = json.loads(SELECTION_PATH.read_text())['chosen']
    record = {'split': 'test', 'chosen_on_val': chosen, **_evaluate_chosen(chosen, CANDIDATES, n_boot)}
    TEST_PATH.write_text(json.dumps(record, indent=2))
    return record


def evaluate_test_5d(n_boot: int = 1000) -> dict:
    """5d 的最终评估：只有 val 上的提升达到 MIN_GAIN_5D 时才允许运行；同时和 5c 的组合做配对比较。
    5d final evaluation: allowed only if the val gain reached MIN_GAIN_5D; also
    compared, paired, with the 5c combination."""
    if not SELECTION_5D_PATH.exists():
        raise FileNotFoundError('run --select5d first')
    sel = json.loads(SELECTION_5D_PATH.read_text())
    if not sel['run_test']:
        raise RuntimeError(f"val gain {sel['gain_over_reference']} < {MIN_GAIN_5D}: by the rule fixed "
                           'in advance, Sonnet is not evaluated on test')
    ref = json.loads(TEST_PATH.read_text())
    assert ref['chosen_on_val'] == REFERENCE_5D
    ref_ids = ref['doc_ids']
    record = {'split': 'test', 'chosen_on_val': sel['chosen'],
              **_evaluate_chosen(sel['chosen'], CANDIDATES_5D, n_boot,
                                 reference={s: ref['results'][f'seed {s}']['prob'] for s in SEEDS})}
    # 参照的概率按 doc_id 顺序保存，这里确认两边顺序一致 / both must use the same statement order
    assert record['doc_ids'] == ref_ids
    TEST_5D_PATH.write_text(json.dumps(record, indent=2))
    return record


def build_speaker_prior(m: float = 10.0) -> dict:
    """说话人名声基线（诊断用的捷径）：说话人在 LIAR *训练集* 中说真话的比例，向总体比例收缩。
    The speaker-reputation shortcut: the speaker's true rate in LIAR TRAIN,
    shrunk towards the overall rate.

    只用训练集的标签，所以 val 和 test 的标签没有泄漏进来（数据集自带的"历史核查次数"包含当前标签，
    不能用）。收缩：(该说话人真陈述数 + m × 总体比例) / (该说话人陈述数 + m)，m=10 事先固定、不调整，
    这样只出现过一两次的说话人不会得到 0 或 1 这样的极端值；训练集里没出现过的说话人得到总体比例。
    Train labels only, so no val / test label leaks in (LIAR's credit-history
    columns include the current label and are not used). Shrinkage with a fixed
    m=10 keeps rare speakers away from 0 or 1; unseen speakers get the overall rate.
    """
    from src.external import load_liar_metadata
    liar = load_liar()
    df = liar[['doc_id', 'label', 'split']].merge(load_liar_metadata()[['doc_id', 'speaker']], on='doc_id')
    train = df[df['split'] == 'train']
    overall = train['label'].mean()
    stats = train.groupby('speaker')['label'].agg(['sum', 'count'])
    prior = (stats['sum'] + m * overall) / (stats['count'] + m)
    out = {}
    for split in ('val', 'test'):
        part = df[df['split'] == split]
        out[f'prob::LIAR {split}'] = part['speaker'].map(prior).fillna(overall).to_numpy()
        out[f'doc_id::LIAR {split}'] = part['doc_id'].to_numpy(dtype=str)
    np.savez_compressed(LLM_DIR / 'speaker_prior.npz', **out)
    return out


def _paths(tag: str):
    f = EXTENSIONS[tag]['file']
    return LLM_DIR / f'combination_{f}_val.json', LLM_DIR / f'combination_{f}_test.json'


def select_extension(tag: str, val: pd.DataFrame = None) -> dict:
    """5f / 5e：只用 val 比较候选，并对每个问题按 MIN_GAIN_5D 决定是否值得在 test 上评估。
    5f / 5e on val only: compare candidates and, for each question, decide by
    MIN_GAIN_5D whether a test evaluation is justified."""
    spec = EXTENSIONS[tag]
    val = load_scores('val', names=_models_needed(spec['candidates'])) if val is None else val
    table = compare_on_val(val, spec['candidates'])
    questions = {}
    for q, (reference, eligible) in spec['questions'].items():
        best = max(eligible, key=lambda c: table[c]['mean CV AUC'])
        gain = round(table[best]['mean CV AUC'] - table[reference]['mean CV AUC'], 4)
        questions[q] = {'reference': reference, 'best': best, 'gain': gain, 'run_test': bool(gain >= MIN_GAIN_5D)}
    corr = val.drop(columns='label').corr(method='spearman').round(2)
    record = {'split': 'val', 'n': len(val), 'step': tag,
              'rule': f'evaluate on test only if a question gains at least {MIN_GAIN_5D} mean 5-fold CV AUC on LIAR val',
              'cv_auc': table, 'questions': questions, 'spearman': corr.to_dict()}
    _paths(tag)[0].write_text(json.dumps(record, indent=2))
    return record


def evaluate_extension(tag: str, n_boot: int = 1000) -> dict:
    """对 val 上达到标准的每个问题，在 test 上评估一次：被选中的组合和它的参照组合都只用 val 训练，
    然后在 test 上做配对比较。没有问题达标时什么也不评估。
    For each question that passed on val, evaluate once on test: the chosen
    combination and its reference are both fitted on val, then compared paired."""
    spec = EXTENSIONS[tag]
    sel_path, test_path = _paths(tag)
    if not sel_path.exists():
        raise FileNotFoundError(f'select {tag} on val first')
    sel = json.loads(sel_path.read_text())
    out = {'split': 'test', 'step': tag, 'questions': {}}
    for q, info in sel['questions'].items():
        if not info['run_test']:
            out['questions'][q] = {'evaluated': False, 'reason': f"val gain {info['gain']} < {MIN_GAIN_5D}"}
            continue
        ref = _evaluate_chosen(info['reference'], spec['candidates'], n_boot)
        ref_probs = {s: ref['results'][f'seed {s}']['prob'] for s in SEEDS}
        res = _evaluate_chosen(info['best'], spec['candidates'], n_boot, reference=ref_probs)
        assert res['doc_ids'] == ref['doc_ids']
        out['doc_ids'] = res['doc_ids']
        out['questions'][q] = {'evaluated': True, 'chosen_on_val': info['best'], 'reference': info['reference'],
                               'reference_auc': {f'seed {s}': ref['results'][f'seed {s}']['combined']['AUC']
                                                 for s in SEEDS},
                               'results': res['results']}
    test_path.write_text(json.dumps(out, indent=2))
    return out


def _print_test(record):
    for seed, r in record['results'].items():
        extra = f" | vs 5c {r['combined - reference']}" if 'combined - reference' in r else ''
        print(seed, r['weights (standardised logits)'], '| combined', r['combined']['AUC'],
              r['combined']['accuracy %'], '| DistilBERT', r['DistilBERT alone']['AUC'],
              '| vs DistilBERT', r['combined - DistilBERT'], extra)


def main():
    parser = argparse.ArgumentParser(description='Steps 5c / 5d: combine text models with the LLM.')
    parser.add_argument('--select', action='store_true', help='5c: compare candidates on LIAR val')
    parser.add_argument('--test', action='store_true', help='5c: evaluate the chosen candidate on LIAR test')
    parser.add_argument('--select5d', action='store_true', help='5d: does Sonnet help, on LIAR val')
    parser.add_argument('--test5d', action='store_true', help='5d: evaluate on LIAR test if the rule allows')
    parser.add_argument('--speaker-prior', action='store_true', help='build the speaker-reputation baseline')
    parser.add_argument('--select-ext', choices=list(EXTENSIONS), help='5f / 5e: compare on LIAR val')
    parser.add_argument('--test-ext', choices=list(EXTENSIONS), help='5f / 5e: evaluate passed questions on test')
    args = parser.parse_args()
    if args.speaker_prior:
        build_speaker_prior()
    if args.select_ext:
        record = select_extension(args.select_ext)
        print(pd.DataFrame(record['cv_auc']).T.sort_values('mean CV AUC', ascending=False).to_string())
        for q, info in record['questions'].items():
            print(q, info)
    if args.test_ext:
        record = evaluate_extension(args.test_ext)
        for q, info in record['questions'].items():
            print('##', q, {k: v for k, v in info.items() if k != 'results'})
            if info.get('evaluated'):
                _print_test(info)
    if args.select:
        record = select()
        print(pd.DataFrame(record['cv_auc']).T.sort_values('mean CV AUC', ascending=False))
        print('chosen on val:', record['chosen'])
    if args.test:
        _print_test(evaluate_test())
    if args.select5d:
        record = select_5d()
        print(pd.DataFrame(record['cv_auc']).T.sort_values('mean CV AUC', ascending=False))
        print('single models, val AUC:', record['single_model_val_auc'])
        print('best with Sonnet:', record['best_with_sonnet'], '| gain over reference:',
              record['gain_over_reference'], '| run on test:', record['run_test'])
    if args.test5d:
        _print_test(evaluate_test_5d())


if __name__ == '__main__':
    main()
