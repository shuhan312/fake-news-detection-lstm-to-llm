"""
实验运行器 / Experiment runner.

每次运行 = 一个模型配置 × 一个随机种子。结果保存为两个文件：
One run = one model config × one seed, saved as two files:
    results/experiments/<model>__<config>__seed<seed>.json   配置、训练信息、各评估集指标
                                                            config, training info, metrics
    results/experiments/<model>__<config>__seed<seed>.npz    每篇文章的"真新闻概率"和 doc_id
                                                            per-article P(real) and doc_ids

已经存在的结果会被跳过，所以中途中断后重新运行同一条命令，会从断点继续。
Existing results are skipped, so re-running the same command resumes.

评估集 / Evaluation sets
    ISOT val       用于 early stopping 和选择配置 / used for early stopping and model choice
    ISOT test      同数据集表现：只评估，*从不* 用于任何选择 / in-domain; scored, NEVER used to choose
    WELFake test   跨数据集表现（05_cross_dataset 的同一批 4,691 篇）/ cross-dataset (same 4,691 articles)

命令行用法 / Command line (from the project root):
    python -m src.experiments --model tfidf
    python -m src.experiments --model lstm --configs A0 A4 --seeds 42 43 44
    python -m src.experiments --model distilbert --seeds 42 43 44
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines
from src.data import PROJECT_ROOT, get_splits, load_dataset
from src.external import load_welfake_external

RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiments'
DEFAULT_SEEDS = (42, 43, 44)
EVAL_SETS = ('ISOT val', 'ISOT test', 'WELFake test')


def load_experiment_data():
    """加载所有实验共用的数据 / Load the data every experiment shares.

    Returns:
        (train, val, eval_sets)：ISOT train / val，以及 {'ISOT test': ..., 'WELFake test': ...}
    """
    isot_train, isot_val, isot_test = get_splits(load_dataset(verbose=False))
    welfake = load_welfake_external()
    welfake_test = welfake[welfake['split'] == 'test'].reset_index(drop=True)
    return isot_train, isot_val, {'ISOT test': isot_test, 'WELFake test': welfake_test}


def result_path(model: str, config: str, seed: int) -> Path:
    return RESULTS_DIR / f'{model}__{config}__seed{seed}'


def _save(model, config_name, seed, config, info, probs, doc_ids):
    """保存一次运行：指标写进 JSON，逐篇概率写进 NPZ。
    Save one run: metrics to JSON, per-article probabilities to NPZ."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(model, config_name, seed)
    metrics = {name: baselines.evaluate(labels, probs[name])
               for name, labels in doc_ids['labels'].items()}
    record = {'model': model, 'config': config_name, 'seed': seed, 'config_detail': config,
              'info': info, 'metrics': metrics,
              'finished_at': datetime.now().isoformat(timespec='seconds'),
              'machine': platform.platform()}
    path.with_suffix('.json').write_text(json.dumps(record, indent=2))
    np.savez_compressed(path.with_suffix('.npz'),
                        **{f'prob::{k}': v for k, v in probs.items()},
                        **{f'doc_id::{k}': np.asarray(v) for k, v in doc_ids['ids'].items()})
    return metrics


def run(model: str, config_names=None, seeds=DEFAULT_SEEDS):
    """依次运行所有（配置, 种子），跳过已完成的。
    Run every (config, seed), skipping those already done."""
    train, val, eval_sets = load_experiment_data()
    all_sets = {'ISOT val': val, **eval_sets}
    doc_ids = {'ids': {k: d['doc_id'].to_numpy() for k, d in all_sets.items()},
               'labels': {k: d['label'].to_numpy() for k, d in all_sets.items()}}

    if model == 'tfidf':
        # TF-IDF 基线是确定性的（同样的输入永远得到同样的结果），跑一次就够
        # The TF-IDF baseline is deterministic, so one run is enough
        C, _ = baselines.select_C(train, val)
        name, seed = f'C{C:g}', 0
        if result_path(model, name, seed).with_suffix('.json').exists():
            print(f'skip {model} {name} (done)')
            return
        start = time.time()
        vec, clf = baselines.fit_tfidf_lr(train, C)
        probs = {k: baselines.predict_proba_real(vec, clf, d) for k, d in all_sets.items()}
        info = {'C': C, 'train_seconds': round(time.time() - start, 1)}
        _report(model, name, seed, _save(model, name, seed, {'C': C}, info, probs, doc_ids))
        return

    if model == 'lstm':
        from src import lstm_model as module
        configs = [module.CONFIGS[n] for n in (config_names or module.CONFIGS)]
    elif model == 'distilbert':
        from src import distilbert_model as module
        configs = [module.CONFIG]
    else:
        raise ValueError(f'unknown model: {model}')

    for config in configs:
        for seed in seeds:
            if result_path(model, config.name, seed).with_suffix('.json').exists():
                print(f'skip {model} {config.name} seed {seed} (done)', flush=True)
                continue
            print(f'\n=== {model} {config.name} seed {seed}: {config.description} ===', flush=True)
            info, probs = module.train_and_predict(config, seed, train, val, eval_sets)
            metrics = _save(model, config.name, seed, module.config_dict(config), info, probs, doc_ids)
            _report(model, config.name, seed, metrics, info)


def _report(model, config, seed, metrics, info=None):
    extra = f" | best epoch {info['best_epoch']}/{info['epochs_run']}, {info['train_seconds']:.0f}s" \
        if info and 'best_epoch' in info else ''
    print(f'{model} {config} seed {seed}: ' + ' | '.join(
        f"{k}: acc {m['accuracy %']}% AUC {m['AUC']}" for k, m in metrics.items()) + extra, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 读取和汇总结果 / Loading and summarising results
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """读取所有已完成的运行，返回长表：每行 = 一次运行 × 一个评估集。
    Load every finished run as a long table: one row per run × evaluation set."""
    rows = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        record = json.loads(path.read_text())
        for eval_set, m in record['metrics'].items():
            rows.append({'model': record['model'], 'config': record['config'],
                         'seed': record['seed'], 'eval set': eval_set,
                         'best epoch': record['info'].get('best_epoch'),
                         'train minutes': round(record['info'].get('train_seconds', 0) / 60, 1),
                         **m})
    return pd.DataFrame(rows)


def load_probs(model: str, config: str, seed: int) -> dict:
    """读取一次运行的逐篇概率：{评估集: (doc_id 数组, 概率数组)}。
    Per-article probabilities of one run: {eval set: (doc_ids, probs)}."""
    data = np.load(result_path(model, config, seed).with_suffix('.npz'), allow_pickle=False)
    names = [k.split('::', 1)[1] for k in data.files if k.startswith('prob::')]
    return {n: (data[f'doc_id::{n}'], data[f'prob::{n}']) for n in names}


def summarize(results: pd.DataFrame, metrics=('accuracy %', 'balanced accuracy %', 'AUC')) -> pd.DataFrame:
    """按（模型, 配置, 评估集）汇总多个种子：均值 ± 标准差，以及种子数量。
    Aggregate seeds per (model, config, eval set): mean ± std and the number of seeds.

    标准差反映"换一个随机种子，结果会差多少"。两个配置的差距如果小于标准差，就不能说哪个更好。
    The std shows how much a result moves with the seed; a gap smaller than
    that cannot be called a real difference.
    """
    grouped = results.groupby(['model', 'config', 'eval set'])
    out = grouped[list(metrics)].agg(['mean', 'std'])
    table = pd.DataFrame(index=out.index)
    for m in metrics:
        digits = 3 if m == 'AUC' else 1
        mean, std = out[(m, 'mean')].round(digits), out[(m, 'std')].fillna(0).round(digits)
        table[m] = mean.astype(str) + ' ± ' + std.astype(str)
    table['seeds'] = grouped.size()
    return table


def main():
    parser = argparse.ArgumentParser(description='Run fake-news experiments.')
    parser.add_argument('--model', required=True, choices=['tfidf', 'lstm', 'distilbert'])
    parser.add_argument('--configs', nargs='*', help='LSTM configs, e.g. A0 A4 (default: all)')
    parser.add_argument('--seeds', nargs='*', type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    run(args.model, args.configs, args.seeds)


if __name__ == '__main__':
    main()
