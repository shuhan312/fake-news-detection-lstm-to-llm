"""
LIAR 基准 / LIAR benchmark.

为什么需要 LIAR / Why LIAR
    ISOT 和 WELFake 的标签都是按 *网站* 打的，模型只要认出来源就能答对。LIAR 的 12,836 条陈述
    由 PolitiFact 编辑 *逐条* 核查，同一个政治人物既有真话也有假话，而且所有样本都是同一种格式
    （一句话）。所以在 LIAR 上，"认出来源"的捷径基本不存在，分数才代表"只看文字判断真假"的能力。
    ISOT / WELFake are labelled per SITE; LIAR is fact-checked per STATEMENT, with
    the same speakers saying both true and false things, all in one format. Its
    scores measure truthfulness detection from text, without the source shortcut.

只使用陈述文本 / Statement text only
    LIAR 还附带说话人、党派、职位和"历史核查次数"（第 9–13 列）。按数据集说明，历史核查次数
    *包含当前这条陈述本身*，把它当特征就是标签泄漏；说话人和党派则是另一种"认来源"的捷径。
    所以这里一律只用陈述文本（load_liar() 的 model_text）。
    LIAR also ships speaker, party, job and credit-history counts (columns 9–13).
    The counts INCLUDE the current statement (label leakage) and speaker / party
    are another source shortcut, so only the statement text is used.

实验 / Experiments (results in results/liar/)
    tfidf_liar        TF-IDF + LR，在 LIAR train 上训练，C 在 LIAR val 上选
    tfidf_isot        第 2 步的 ISOT TF-IDF 模型直接用在 LIAR 上（格式差别大，只作参考）
    distilbert_seedN  DistilBERT，在 LIAR train 上微调，max_length 64（覆盖 99.95% 的陈述）

命令行 / Command line:
    python -m src.liar --part tfidf
    python -m src.liar --part distilbert --seeds 42 43 44
"""

import argparse
import json
from dataclasses import replace

import numpy as np

from src import baselines as B
from src.data import PROJECT_ROOT
from src.external import load_liar

RESULTS_DIR = PROJECT_ROOT / 'results' / 'liar'

# LIAR 的陈述很短：DistilBERT 分词后中位数 20 个 token，99% 不超过 45，最长 77。
# 64 覆盖 99.95% 的陈述，比 ISOT 用的 128 快约一倍。
# Statements are short (median 20 tokens, 99th percentile 45, max 77); 64 covers
# 99.95% and is about twice as fast as ISOT's 128.
LIAR_MAX_LENGTH = 64

# LIAR 的 C 搜索范围向下扩展：第一次运行时，默认范围（0.25–256）选中了最小的 0.25，落在边界上。
# LIAR 很小（1 万条、每条十几个词），需要更强的正则化（更小的 C）。
# The C grid extends downwards: with the default grid (0.25–256) LIAR picked
# the smallest value, 0.25 — on the edge. LIAR is small (10k short statements)
# and needs stronger regularisation (smaller C).
# 扩展后 0.25 仍然最好，并且落在范围内部（0.0625 和 1.0 都更差），说明调参充分。
# With the extended grid 0.25 is still best and now interior (0.0625 and 1.0 are worse).
LIAR_C_GRID = (0.004, 0.016, 0.0625, 0.25, 1.0, 4.0)


def _splits():
    liar = load_liar()
    return tuple(liar[liar['split'] == s].reset_index(drop=True) for s in ('train', 'val', 'test'))


def _save(name: str, eval_sets: dict, probs: dict, extra: dict):
    """保存指标（JSON）和逐条概率（NPZ）/ Save metrics (JSON) and per-statement probabilities (NPZ)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {k: B.evaluate(df['label'], probs[k]) for k, df in eval_sets.items()}
    path = RESULTS_DIR / name
    path.with_suffix('.json').write_text(json.dumps({'name': name, **extra, 'metrics': metrics}, indent=2))
    np.savez_compressed(path.with_suffix('.npz'),
                        **{f'prob::{k}': probs[k] for k in eval_sets},
                        **{f'doc_id::{k}': df['doc_id'].to_numpy(dtype=str) for k, df in eval_sets.items()})
    print(f'saved {name}:', {k: (m['accuracy %'], m['AUC']) for k, m in metrics.items()}, flush=True)


def run_tfidf():
    """TF-IDF 基准：在 LIAR 上训练，以及把 ISOT 模型直接用在 LIAR 上。
    TF-IDF baselines: trained on LIAR, and the ISOT model applied to LIAR."""
    train, val, test = _splits()
    eval_sets = {'LIAR val': val, 'LIAR test': test}

    C, table = B.select_C(train, val, grid=LIAR_C_GRID)
    vec, clf = B.fit_tfidf_lr(train, C)
    _save('tfidf_liar', eval_sets, {k: B.predict_proba_real(vec, clf, d) for k, d in eval_sets.items()},
          {'C': C, 'C_table': table.reset_index().to_dict('records')})

    # ISOT 模型 → LIAR：ISOT 是几百词的整篇文章，LIAR 是一句话，格式差别很大，结果只作参考
    # ISOT model → LIAR: whole articles vs one sentence, so this is only indicative
    # （C 在 ISOT 上选择，和 LIAR 的范围无关；已完成就跳过 / C is chosen on ISOT; skipped if done）
    if (RESULTS_DIR / 'tfidf_isot.json').exists():
        print('skip tfidf_isot (done)', flush=True)
        return
    from src.data import get_splits, load_dataset
    isot_train, isot_val, _ = get_splits(load_dataset(verbose=False))
    C_isot, _ = B.select_C(isot_train, isot_val)
    vec, clf = B.fit_tfidf_lr(isot_train, C_isot)
    _save('tfidf_isot', eval_sets, {k: B.predict_proba_real(vec, clf, d) for k, d in eval_sets.items()},
          {'C': C_isot, 'trained_on': 'ISOT train'})


def run_distilbert(seeds=(42, 43, 44)):
    """DistilBERT 在 LIAR 上微调。其他设置和 notebook 03 相同，只把 max_length 改为 64。
    DistilBERT fine-tuned on LIAR; notebook 03 settings except max_length 64.
    已完成的种子会跳过 / finished seeds are skipped."""
    from src import distilbert_model as D

    train, val, test = _splits()
    config = replace(D.CONFIG, name='DistilBERT-LIAR', max_length=LIAR_MAX_LENGTH,
                     description=f'distilbert-base-uncased on LIAR, max_length {LIAR_MAX_LENGTH}')
    for seed in seeds:
        name = f'distilbert_seed{seed}'
        if (RESULTS_DIR / name).with_suffix('.json').exists():
            print(f'skip {name} (done)', flush=True)
            continue
        print(f'\n=== LIAR {name} ===', flush=True)
        info, probs = D.train_and_predict(config, seed, train, val, {'LIAR test': test})
        # train_and_predict 把验证集命名为 'ISOT val'，这里改回 'LIAR val'
        # train_and_predict names the validation set 'ISOT val'; rename it
        probs['LIAR val'] = probs.pop('ISOT val')
        _save(name, {'LIAR val': val, 'LIAR test': test}, probs,
              {'seed': seed, 'config': D.config_dict(config), 'info': info})


def main():
    parser = argparse.ArgumentParser(description='LIAR benchmark.')
    parser.add_argument('--part', required=True, choices=['tfidf', 'distilbert'])
    parser.add_argument('--seeds', nargs='*', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    run_tfidf() if args.part == 'tfidf' else run_distilbert(args.seeds)


if __name__ == '__main__':
    main()
