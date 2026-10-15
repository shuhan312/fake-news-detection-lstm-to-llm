"""
第 4 步：降低模型对"来源文风"的依赖 / Step 4: reducing reliance on source style.

三种方法，都先用 TF-IDF + 逻辑回归（几分钟一次），找出有效的再用 DistilBERT 确认。
Three methods, first with TF-IDF + logistic regression (minutes per run).

    4a 只保留内容词   content words only    训练前删掉虚词和文风标记（"said"、星期几、"Mr." 等）
    4b 去偏重加权     de-biased reweighting 先训练一个只看文风词的"偏置模型"，正式训练时降低
                                            它能轻松判对的文章的权重
    4c 多来源训练     multi-source training ISOT + WELFake 的一部分来源一起训练，在训练时
                                            *完全没见过* 的出版方上测试

评估规则（避免"用测试集选方法"）/ Evaluation rules (no test-set selection):
    - 每个模型的 C：在 *训练数据* 的验证集上选（和 src/baselines.select_C 相同）
      C is chosen on the TRAINING data's validation split
    - 比较、挑选方法：用 WELFake *val*；WELFake *test* 只在最后报告
      methods are compared on WELFake VAL; WELFake TEST is reported only at the end
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src import baselines as B

# ─────────────────────────────────────────────────────────────────────────────
# 4a：内容词 / Content words
# ─────────────────────────────────────────────────────────────────────────────

# 文风标记：在 ISOT 里区分路透社和博客、但和内容真假无关的词。
# 来源：04_leakage_audit 和 05_cross_dataset 中权重最大、而且换数据集后方向会翻转的特征。
# Style markers: words that separate Reuters from blogs in ISOT but say nothing
# about truthfulness — taken from the top / sign-flipping features found in
# 04_leakage_audit and 05_cross_dataset.
STYLE_MARKERS = frozenset(
    # 引述动词 / reporting verbs
    'said says say saying told tell tells according added stated statement statements '
    'spokesman spokeswoman spokesperson reporters reported reporting '
    # 星期和月份缩写（路透社写 "on Wednesday"、"Nov. 8"）/ weekdays and month abbreviations
    'monday tuesday wednesday thursday friday saturday sunday '
    'jan feb mar apr jun jul aug sep sept oct nov dec '
    # 称谓缩写（《纽约时报》写 "Mr."，路透社不写）/ honorifics
    'mr mrs ms dr sen rep gov '
    # 博客口吻的强调和指示词 / blog-voice intensifiers and deixis
    'just even really actually literally here watch video'.split())

# sklearn 的英文停用词表（318 个，来自 Glasgow 停用词表）里混进了不少 *实词*，例如 "bill"（法案）、
# "fire"、"interest"、"system"，以及数字。新闻里这些是重要的内容词，不能当作虚词删掉，
# 所以从停用词表中去掉它们（单元测试 test_content_and_style_views_are_exact_complements 发现了这个问题）。
# sklearn's stop-word list (318 words, from the Glasgow list) contains several CONTENT
# words — "bill", "fire", "interest", "system", numbers — that matter in news, so
# they are removed from it (found by a unit test).
CONTENT_WORDS_IN_SKLEARN_LIST = frozenset(
    'bill fire interest system call find found give move name part put see show side take keep made '
    'top bottom front back full empty fill detail describe serious sincere thick thin mill con cry de '
    'un co inc ltd amount former formerly first third last one two three four five six eight nine ten '
    'eleven twelve fifteen twenty forty fifty sixty hundred'.split())
FUNCTION_WORDS = frozenset(ENGLISH_STOP_WORDS) - CONTENT_WORDS_IN_SKLEARN_LIST

# 注意："us" 留在虚词表里。model_text 把 "U.S." 合并成了 "us"，所以这个词同时代表代词 "us" 和
# 国家 "U.S."，无法区分；它在 ISOT 和 WELFake 中的方向正好相反（见 05_cross_dataset），按文风词处理。
# Note: "us" stays a function word. model_text merges "U.S." into "us", so the
# token mixes the pronoun and the country; its direction flips between ISOT and
# WELFake (05_cross_dataset), so it is treated as a style word.

# 虚词 + 文风标记 / Function words + style markers
STYLE_WORDS = FUNCTION_WORDS | STYLE_MARKERS

# 第 4 步只搜索这三个 C：05 和第 2 步中，所有 TF-IDF 模型选出的最佳 C 都在 16–64 之间，
# 这样可以把每次运行的时间减半。
# Step 4 searches only these: every TF-IDF model in 05 / step 2 picked C in 16–64,
# so this halves the run time.
STEP4_C_GRID = (4.0, 16.0, 64.0)


def content_only(texts: pd.Series, remove=STYLE_WORDS) -> pd.Series:
    """删掉虚词和文风标记，只保留内容词（输入是已经小写、去标点的 model_text）。
    Drop function words and style markers, keeping content words (input: model_text)."""
    return texts.str.split().map(lambda words: ' '.join(w for w in words if w not in remove))


def style_only(texts: pd.Series, keep=STYLE_WORDS) -> pd.Series:
    """只保留虚词和文风标记 —— 和 content_only 正好相反，用作 4b 的"偏置模型"输入。
    Keep ONLY function words and style markers — the complement of content_only,
    used as the input of 4b's bias model."""
    return texts.str.split().map(lambda words: ' '.join(w for w in words if w in keep))


# ─────────────────────────────────────────────────────────────────────────────
# 通用：在任意文本视角上训练和评估 / Train and score on any text view
# ─────────────────────────────────────────────────────────────────────────────

def fit_view(train: pd.DataFrame, val: pd.DataFrame, view=lambda s: s, sample_weight=None):
    """在某个"文本视角"上训练 TF-IDF + 逻辑回归，C 在 val 上按 AUC 选择。
    Fit TF-IDF + LR on a text view; C chosen by AUC on `val` (the training data's own val).

    Args:
        view: 把 model_text 转换成模型实际读取的文本（例如 content_only）。
              maps model_text to the text the model reads (e.g. content_only).
        sample_weight: 每篇训练文章的权重（4b 用）；None = 全部相等。
                       per-article training weights (4b); None = equal.

    Returns:
        predict(df) -> 真新闻概率的函数，以及选中的 C
        a predict(df) -> P(real) function, and the chosen C
    """
    x_train_text, x_val_text = view(train['model_text']), view(val['model_text'])
    best = None
    for C in STEP4_C_GRID:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, sublinear_tf=True)
        x_train = vec.fit_transform(x_train_text)
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(x_train, train['label'], sample_weight=sample_weight)
        auc = roc_auc_score(val['label'], clf.predict_proba(vec.transform(x_val_text))[:, 1])
        # 并列时保留较小的 C（正则化更强）/ ties keep the smaller C (stronger regularisation)
        if best is None or round(auc, 4) > round(best[0], 4):
            best = (auc, C, vec, clf)
    _, C, vec, clf = best

    def predict(df):
        return clf.predict_proba(vec.transform(view(df['model_text'])))[:, 1]
    predict.vectorizer, predict.classifier = vec, clf
    return predict, C


# ─────────────────────────────────────────────────────────────────────────────
# 4b：去偏重加权 / De-biased reweighting
# ─────────────────────────────────────────────────────────────────────────────

def bias_weights(train: pd.DataFrame, C: float = 4.0, gamma: float = 1.0, folds: int = 5, seed: int = 42):
    """计算 4b 的训练权重：偏置模型越有把握判对的文章，权重越低。
    Training weights for 4b: the more confidently the bias model is right, the lower the weight.

    步骤 / Steps
        1. 偏置模型 = 只看虚词和文风标记（style_only）的 TF-IDF + 逻辑回归
           bias model = TF-IDF + LR on style words only
        2. 用 5 折交叉验证得到每篇训练文章的预测：每篇文章的预测都来自 *没见过它* 的模型。
           如果直接在训练集上训练再预测，偏置模型会把训练文章"背下来"，几乎全部判对，权重就失去意义。
           5-fold out-of-fold predictions: each article is scored by a model that
           never saw it. Scoring the training set directly would let the bias model
           memorise it and get nearly everything right, making the weights meaningless.
        3. 权重 = (1 − 偏置模型给正确标签的概率) ^ gamma，再缩放到平均值为 1。
           gamma 越大，对"靠文风就能判对"的文章惩罚越重。
           weight = (1 − P_bias(true label)) ^ gamma, rescaled to mean 1; a larger
           gamma penalises style-solvable articles more.

    Returns:
        (weights, p_true)：每篇文章的权重，以及偏置模型给正确标签的概率（用于分析）
    """
    x = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, sublinear_tf=True) \
        .fit_transform(style_only(train['model_text']))
    prob_real = cross_val_predict(LogisticRegression(C=C, max_iter=2000), x, train['label'],
                                  cv=folds, method='predict_proba')[:, 1]
    y = train['label'].to_numpy()
    p_true = np.where(y == 1, prob_real, 1 - prob_real)
    w = (1 - p_true) ** gamma
    # 加一个很小的下限，避免权重恰好为 0 的文章被完全丢弃 / tiny floor so no article gets weight 0
    w = np.maximum(w, 1e-3)
    return w / w.mean(), p_true


# ─────────────────────────────────────────────────────────────────────────────
# 评估 / Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def score(predict, eval_sets: dict) -> dict:
    """在多个评估集上计算指标 / Metrics on several evaluation sets."""
    return {name: B.evaluate(df['label'], predict(df)) for name, df in eval_sets.items()}


def results_table(rows: dict, metrics=('accuracy %', 'balanced accuracy %', 'AUC')) -> pd.DataFrame:
    """{方法名: score() 的结果} → 一张表，行是（方法, 评估集）。
    {method: score() output} → one table indexed by (method, eval set)."""
    return pd.concat({method: pd.DataFrame(sets).T[list(metrics)] for method, sets in rows.items()})


# ─────────────────────────────────────────────────────────────────────────────
# 4c：多来源训练，留出一个出版方 / Multi-source training with a held-out outlet
# ─────────────────────────────────────────────────────────────────────────────

def leave_outlet_out(isot_splits, welfake: pd.DataFrame, groups: pd.Series, held_out: str):
    """构造"留出一个出版方"的训练 / 验证 / 测试数据。
    Build train / val / test data that leave one real-news outlet out.

    Args:
        isot_splits: (ISOT train, ISOT val)
        welfake: load_welfake_external() 的全部数据（含 split 列）/ all of WELFake-external
        groups: src.analysis.publisher_groups(welfake)，以 doc_id 为索引
        held_out: 被留出的出版方，例如 'The New York Times' / the outlet to hold out

    Returns:
        dict，包含 / with keys:
            'train', 'val'  ISOT + WELFake 的同名划分，*去掉* 被留出出版方的全部文章
                            ISOT + WELFake splits WITHOUT any article from the held-out outlet
            'welfake_train', 'welfake_val'  只有 WELFake 部分（用于"只用 WELFake 训练"的对照）
            'test'  WELFake test 中：被留出出版方的真新闻 + 全部假新闻
                    WELFake test: the held-out outlet's real articles + all fake articles

    说明：测试集里的 *真新闻* 来自训练时完全没见过的出版方；假新闻来自和训练集相同的
    那批网站（WELFake 没有提供假新闻网站的名称，无法按网站留出）。
    Note: the test REAL articles come from an outlet never seen in training; the
    fake articles come from the same sites as training (WELFake does not name
    its fake sites, so they cannot be held out).
    """
    isot_train, isot_val = isot_splits
    held = groups.loc[welfake['doc_id']].to_numpy() == f'REAL: {held_out}'
    seen = welfake[~held]
    welfake_train = seen[seen['split'] == 'train']
    welfake_val = seen[seen['split'] == 'val']
    test_mask = (welfake['split'] == 'test') & (held | (welfake['label'] == 0))
    return {
        'train': pd.concat([isot_train, welfake_train], ignore_index=True),
        'val': pd.concat([isot_val, welfake_val], ignore_index=True),
        'welfake_train': welfake_train.reset_index(drop=True),
        'welfake_val': welfake_val.reset_index(drop=True),
        'test': welfake[test_mask].reset_index(drop=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 运行与保存 / Running and saving
#
#   python -m src.step4 --part 4ab     4a + 4b（只用 ISOT 训练，在 WELFake 上测试）
#   python -m src.step4 --part 4c      4c（留出《纽约时报》/ Breitbart）
# ─────────────────────────────────────────────────────────────────────────────

def _results_dir():
    from src.data import PROJECT_ROOT
    path = PROJECT_ROOT / 'results' / 'step4'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(name: str, metrics: dict, extra: dict, probs: dict):
    """保存一个方法的指标（JSON）和逐篇概率（NPZ，doc_id 存为字符串）。
    Save one method's metrics (JSON) and per-article probabilities (NPZ)."""
    import json
    path = _results_dir() / name
    path.with_suffix('.json').write_text(json.dumps({'name': name, **extra, 'metrics': metrics}, indent=2))
    np.savez_compressed(path.with_suffix('.npz'),
                        **{f'prob::{k}': p for k, (ids, p) in probs.items()},
                        **{f'doc_id::{k}': np.asarray(ids, dtype=str) for k, (ids, p) in probs.items()})
    print(f'saved {name}:', {k: (v['accuracy %'], v['AUC']) for k, v in metrics.items()}, flush=True)


def _run_and_save(name, predict, extra, eval_sets):
    probs = {k: (df['doc_id'].to_numpy(), predict(df)) for k, df in eval_sets.items()}
    metrics = {k: B.evaluate(df['label'], probs[k][1]) for k, df in eval_sets.items()}
    _save(name, metrics, extra, probs)


def run_4ab():
    """4a 和 4b：只用 ISOT 训练，在 ISOT val、WELFake val、WELFake test 上评估。
    4a and 4b: train on ISOT only; score ISOT val, WELFake val and WELFake test."""
    from src.data import get_splits, load_dataset
    from src.external import load_welfake_external

    isot_train, isot_val, _ = get_splits(load_dataset(verbose=False))
    welfake = load_welfake_external()
    eval_sets = {'ISOT val': isot_val,
                 'WELFake val': welfake[welfake['split'] == 'val'],
                 'WELFake test': welfake[welfake['split'] == 'test']}

    # 对照：完整文本 / reference: full text
    predict, C = fit_view(isot_train, isot_val)
    _run_and_save('baseline_full_text', predict, {'C': C}, eval_sets)

    # 4a：只保留内容词；以及反过来，只保留文风词（用来看文风本身能做到多少）
    # 4a: content words only; and the reverse, style words only
    for name, view in [('4a_content_only', content_only), ('style_only', style_only)]:
        predict, C = fit_view(isot_train, isot_val, view)
        _run_and_save(name, predict, {'C': C}, eval_sets)

    # 4b：去偏重加权，gamma = 1 和 2 / de-biased reweighting with gamma 1 and 2
    for gamma in (1.0, 2.0):
        weights, p_true = bias_weights(isot_train, gamma=gamma)
        predict, C = fit_view(isot_train, isot_val, sample_weight=weights)
        extra = {'C': C, 'gamma': gamma,
                 'bias_model_oof_accuracy': float((p_true > 0.5).mean()),
                 'weight_quantiles': {q: float(np.quantile(weights, q)) for q in (0.1, 0.5, 0.9, 0.99)}}
        _run_and_save(f'4b_reweight_gamma{gamma:g}', predict, extra, eval_sets)


def run_4c(outlets=('The New York Times', 'Breitbart')):
    """4c：多来源训练，每次留出一个真新闻出版方，只在它上面测试。
    4c: multi-source training, holding out one real-news outlet at a time."""
    from src.analysis import publisher_groups
    from src.data import get_splits, load_dataset
    from src.external import load_welfake_external

    isot_train, isot_val, _ = get_splits(load_dataset(verbose=False))
    welfake = load_welfake_external()
    groups = publisher_groups(welfake)

    # 只用 ISOT 训练的模型不依赖留出的出版方，训练一次即可 / the ISOT-only model is trained once
    isot_only, C_isot = fit_view(isot_train, isot_val)
    for outlet in outlets:
        tag = 'nyt' if 'Times' in outlet else outlet.lower()
        data = leave_outlet_out((isot_train, isot_val), welfake, groups, outlet)
        test = {f'held-out {outlet} + fakes': data['test']}
        extra = {'held_out': outlet, 'n_train': len(data['train']),
                 'n_test_real': int(data['test']['label'].sum()),
                 'n_test_fake': int((data['test']['label'] == 0).sum())}

        _run_and_save(f'4c_{tag}_isot_only', isot_only, {**extra, 'C': C_isot}, test)
        predict, C = fit_view(data['welfake_train'], data['welfake_val'])
        _run_and_save(f'4c_{tag}_welfake_only', predict, {**extra, 'C': C}, test)
        predict, C = fit_view(data['train'], data['val'])
        _run_and_save(f'4c_{tag}_isot_plus_welfake', predict, {**extra, 'C': C}, test)
        predict, C = fit_view(data['train'], data['val'], content_only)
        _run_and_save(f'4c_{tag}_isot_plus_welfake_content_only', predict, {**extra, 'C': C}, test)


def run_distilbert(method: str, seeds=(42, 43, 44), gamma: float = 1.0,
                   outlet: str = 'The New York Times'):
    """用 DistilBERT 确认 TF-IDF 上效果最好的第 4 步方法（每个种子约 1.5–3 小时）。
    Confirm the best step-4 method with DistilBERT (~1.5–3 h per seed).

    method:
        '4a'  输入换成 content_only 文本，只用 ISOT 训练 / content-only input, ISOT training
        '4b'  只用 ISOT 训练，按 bias_weights(gamma) 加权 / ISOT training with bias weights
        '4c'  ISOT + WELFake（留出 outlet）训练，在留出的出版方上测试 / multi-source, held-out outlet
        'none' 不做任何改动的 DistilBERT，用作 4c 同一测试集上的对照（4a / 4b 的对照就是第 2 步的结果）
               plain DistilBERT, the reference on 4c's test set
    结果保存为 results/step4/distilbert_<method>_seed<seed>.json / .npz；已完成的种子会跳过。
    Results go to results/step4/; finished seeds are skipped.
    """
    from src import distilbert_model as D
    from src.analysis import publisher_groups
    from src.data import get_splits, load_dataset
    from src.external import load_welfake_external

    isot_train, isot_val, _ = get_splits(load_dataset(verbose=False))
    welfake = load_welfake_external()
    welfake_eval = {'WELFake val': welfake[welfake['split'] == 'val'],
                    'WELFake test': welfake[welfake['split'] == 'test']}
    weights, extra = None, {'method': method}

    if method == '4a':
        # 所有数据都换成"只保留内容词"的版本 / every split gets the content-only text
        swap = lambda df: df.assign(model_text=content_only(df['model_text']))
        train, val = swap(isot_train), swap(isot_val)
        eval_sets = {k: swap(v) for k, v in welfake_eval.items()}
    elif method == '4b':
        train, val, eval_sets = isot_train, isot_val, welfake_eval
        weights, _ = bias_weights(isot_train, gamma=gamma)
        extra['gamma'] = gamma
    elif method in ('4c', 'none'):
        data = leave_outlet_out((isot_train, isot_val), welfake, publisher_groups(welfake), outlet)
        train, val = (data['train'], data['val']) if method == '4c' else (isot_train, isot_val)
        eval_sets = {f'held-out {outlet} + fakes': data['test'], **welfake_eval}
        extra['held_out'] = outlet
    else:
        raise ValueError(f'unknown method: {method}')

    for seed in seeds:
        name = f'distilbert_{method}_seed{seed}'
        if (_results_dir() / name).with_suffix('.json').exists():
            print(f'skip {name} (done)', flush=True)
            continue
        print(f'\n=== {name} ===', flush=True)
        info, probs = D.train_and_predict(D.CONFIG, seed, train, val, eval_sets, sample_weight=weights)
        # train_and_predict 总是把验证集的预测命名为 'ISOT val'；4c 的验证集是 ISOT + WELFake 混合的，
        # 所以改名为 'mixed val'，避免和真正的 ISOT val 混淆
        # train_and_predict always names the validation predictions 'ISOT val'; 4c's
        # validation set mixes ISOT and WELFake, so it is renamed 'mixed val'
        val_name = 'mixed val' if method == '4c' else 'ISOT val'
        probs[val_name] = probs.pop('ISOT val')
        all_sets = {val_name: val, **eval_sets}
        metrics = {k: B.evaluate(df['label'], probs[k]) for k, df in all_sets.items()}
        _save(name, metrics, {**extra, 'seed': seed, 'info': info},
              {k: (df['doc_id'].to_numpy(), probs[k]) for k, df in all_sets.items()})


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 4 experiments.')
    parser.add_argument('--part', required=True, choices=['4ab', '4c', 'distilbert'])
    parser.add_argument('--method', choices=['4a', '4b', '4c', 'none'], help='for --part distilbert')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--seeds', nargs='*', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    if args.part == '4ab':
        run_4ab()
    elif args.part == '4c':
        run_4c()
    else:
        run_distilbert(args.method, args.seeds, gamma=args.gamma)


if __name__ == '__main__':
    main()
