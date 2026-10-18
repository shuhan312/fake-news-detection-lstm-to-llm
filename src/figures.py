"""
README 的总结图 / The summary figure for the README.

从已保存的结果文件生成，不重新训练任何模型：
    左图：跨来源测试（WELFake 的 1,501 篇文章）上各模型的 AUC
    右图：LIAR test 上各模型的 AUC
Built from saved results only (no retraining):
    left  - AUC on the cross-source test (1,501 WELFake articles)
    right - AUC on LIAR test

命令行 / Command line:
    python -m src.figures      # 写入 results/figures/summary.png / writes results/figures/summary.png
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data import PROJECT_ROOT  # noqa: E402

LLM = PROJECT_ROOT / 'results' / 'llm'
LIAR = PROJECT_ROOT / 'results' / 'liar'
OUT = PROJECT_ROOT / 'results' / 'figures' / 'summary.png'

TRAINED = '#9aa5b1'      # 在 ISOT / LIAR 上训练的模型 / models trained on ISOT or LIAR
LLM_COLOUR = '#3d6fb6'   # 零样本大模型 / zero-shot LLMs
COMBINED = '#1f3f73'     # 组合模型 / combination


def welfake_aucs() -> dict:
    """同一批 1,501 篇 WELFake 文章上的 AUC（ISOT 模型取 3 个种子的平均概率，和 notebook 11 相同）。
    AUC on the same 1,501 WELFake articles (ISOT models: 3-seed mean, as in notebook 11)."""
    from src.experiments import load_probs
    from src.llm_crosssource import sample_test
    sample = sample_test().set_index('doc_id')
    ids, y = sample.index, sample['label']

    def isot(model, config, seeds):
        runs = []
        for s in seeds:
            doc_ids, prob = load_probs(model, config, s)['WELFake test']
            runs.append(pd.Series(prob, index=np.asarray(doc_ids).astype(str)).loc[ids])
        return pd.concat(runs, axis=1).mean(axis=1)

    def llm(model):
        d = np.load(LLM / f'llm_welfake_{model}.npz')
        return pd.Series(d['prob'], index=d['doc_id']).loc[ids]

    scores = {'LSTM\n(my 2023 paper)': isot('lstm', 'A0', (42, 43, 44)),
              'BiLSTM\n(tuned)': isot('lstm', 'A6', (42, 43, 44)),
              'DistilBERT': isot('distilbert', 'DistilBERT', (42, 43, 44)),
              'TF-IDF': isot('tfidf', 'C64', (0,)),
              'Claude\nHaiku\n(no training)': llm('haiku'),
              'Claude\nSonnet\n(no training)': llm('sonnet')}
    return {k: roc_auc_score(y, v) for k, v in scores.items()}


def liar_aucs() -> dict:
    """LIAR test 上的 AUC，全部从已保存的指标中读取；有 3 个种子的取平均。
    LIAR test AUC from saved metrics; 3-seed runs are averaged."""
    read = lambda p: json.loads(p.read_text())
    seeds = (42, 43, 44)
    final = read(LLM / 'combination_sonnet_test.json')['results']
    return {'TF-IDF': read(LIAR / 'tfidf_liar.json')['metrics']['LIAR test']['AUC'],
            'DistilBERT': np.mean([read(LIAR / f'distilbert_seed{s}.json')['metrics']['LIAR test']['AUC'] for s in seeds]),
            'Claude\nHaiku': read(LLM / 'llm_no_evidence.json')['metrics']['LIAR test']['AUC'],
            'Claude\nSonnet': read(LLM / 'llm_no_evidence_sonnet.json')['metrics']['LIAR test']['AUC'],
            'Text models\n+ Haiku\n+ Sonnet': np.mean([final[f'seed {s}']['combined']['AUC'] for s in seeds])}


def _bars(ax, values: dict, colours: list, title: str, note: str):
    names = list(values)
    bars = ax.bar(range(len(names)), [values[n] for n in names], color=colours, width=0.68)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, values[n] + 0.008, f'{values[n]:.2f}', ha='center', fontsize=9)
    ax.axhline(0.5, color='black', lw=0.8, ls='--')
    ax.text(-0.45, 0.505, 'chance', ha='left', va='bottom', fontsize=8)
    ax.set_xticks(range(len(names)), names, fontsize=8.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel('AUC')
    ax.set_title(title, fontsize=11, loc='left')
    ax.text(0, -0.30, note, transform=ax.transAxes, fontsize=8, color='#444444', va='top')
    ax.spines[['top', 'right']].set_visible(False)


def main():
    welfake, liar = welfake_aucs(), liar_aucs()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [6, 5], 'wspace': 0.25})
    _bars(axes[0], welfake, [TRAINED] * 4 + [LLM_COLOUR] * 2,
          'A. The original problem: news from outlets other than Reuters',
          '1,501 WELFake articles from outlets other than Reuters.\n'
          'Grey: trained on ISOT (93-99% accuracy in-domain). Blue: no training.\n'
          'Caveat: these 2016-17 articles may be in the LLMs\' training data.')
    _bars(axes[1], liar, [TRAINED] * 2 + [LLM_COLOUR] * 2 + [COMBINED],
          'B. Fact-checked statements (LIAR): no source to recognise',
          '1,283 PolitiFact statements (LIAR test).\n'
          'Grey: trained on LIAR. Blue: no training.\n'
          'Combination fitted and chosen on the validation split only.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches='tight')
    print('wrote', OUT, '\nWELFake:', {k.replace(chr(10), ' '): round(v, 3) for k, v in welfake.items()},
          '\nLIAR:', {k.replace(chr(10), ' '): round(v, 3) for k, v in liar.items()})


if __name__ == '__main__':
    main()
