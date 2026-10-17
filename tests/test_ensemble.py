"""
第 5c 步组合器的单元测试（用合成数据，不读结果文件）/ Tests for the step-5c combiner (synthetic data).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ensemble  # noqa: E402


def _synthetic(n=600, seed=0):
    # 两个"模型"各自看到真相的一部分，噪声互相独立 / two models, each seeing part of the truth, independent noise
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return pd.DataFrame({'label': y,
                         'a': sig(1.0 * (y - 0.5) * 2 + rng.normal(0, 1.5, n)),
                         'b': sig(1.0 * (y - 0.5) * 2 + rng.normal(0, 1.5, n)),
                         'noise': rng.uniform(0, 1, n)})


def test_to_logit_is_finite_for_0_and_100_percent():
    # Haiku 会回答 0 或 100 / Haiku can answer 0 or 100
    out = ensemble.to_logit([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(out))
    assert out[1] == 0 and out[0] == pytest.approx(-out[2])


def test_cv_auc_rewards_signal_and_not_noise():
    df = _synthetic()
    assert ensemble.cv_auc(df, ['noise']) < 0.6
    assert ensemble.cv_auc(df, ['a']) > 0.7


def test_combining_independent_models_beats_each_one():
    df = _synthetic(n=2000)
    both = ensemble.cv_auc(df, ['a', 'b'])
    assert both > ensemble.cv_auc(df, ['a']) + 0.02
    assert both > ensemble.cv_auc(df, ['b']) + 0.02


def test_columns_for_replaces_distilbert_with_the_seed_column():
    assert ensemble.columns_for('DistilBERT + Haiku', 43) == ['distilbert_seed43', 'haiku']
    assert ensemble.columns_for('TF-IDF', 42) == ['tfidf']


def test_load_scores_aligns_by_doc_id_and_rejects_missing(tmp_path, monkeypatch):
    liar = pd.DataFrame({'doc_id': ['x1', 'x2', 'x3'], 'label': [1, 0, 1], 'split': 'val'})
    monkeypatch.setattr(ensemble, 'load_liar', lambda: liar)
    # 文件里的顺序和 LIAR 不同：必须按 doc_id 对齐 / stored in a different order: must align by doc_id
    np.savez(tmp_path / 'm.npz', **{'prob::LIAR val': np.array([0.3, 0.1, 0.9]),
                                    'doc_id::LIAR val': np.array(['x3', 'x1', 'x2'])})
    out = ensemble.load_scores('val', {'m': {'val': (tmp_path / 'm.npz', 'LIAR val')}})
    assert out.loc['x1', 'm'] == 0.1 and out.loc['x2', 'm'] == 0.9 and out.loc['x3', 'm'] == 0.3
    # 缺少一条陈述时报错，而不是悄悄错配 / a missing statement raises instead of mis-pairing
    np.savez(tmp_path / 'short.npz', **{'prob::LIAR val': np.array([0.3, 0.1]),
                                        'doc_id::LIAR val': np.array(['x3', 'x1'])})
    with pytest.raises(ValueError):
        ensemble.load_scores('val', {'m': {'val': (tmp_path / 'short.npz', 'LIAR val')}})


def _val_for_5d(sonnet_signal: float, n=1500, seed=1):
    # 文字模型共享同一份噪声（像 TF-IDF 和 DistilBERT 一样高度相关），Haiku 和"Sonnet"各自独立
    # Text models share one noise (highly correlated, like TF-IDF and DistilBERT); Haiku and "Sonnet" are independent
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    sig = lambda z: 1 / (1 + np.exp(-z))
    text_noise = rng.normal(0, 1.5, n)
    df = pd.DataFrame({'label': y, 'tfidf': sig((y - 0.5) * 2 + text_noise + rng.normal(0, 0.3, n))})
    for s in ensemble.SEEDS:
        df[f'distilbert_seed{s}'] = sig((y - 0.5) * 2 + text_noise + rng.normal(0, 0.3, n))
    df['haiku'] = sig((y - 0.5) * 2 + rng.normal(0, 1.5, n))
    df['sonnet'] = sig((y - 0.5) * 2 * sonnet_signal + rng.normal(0, 1.5, n))
    return df


def test_5d_rule_blocks_test_when_sonnet_adds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble, 'SELECTION_5D_PATH', tmp_path / 'sel.json')
    record = ensemble.select_5d(_val_for_5d(sonnet_signal=0.0))      # Sonnet 是纯噪声 / pure noise
    assert record['run_test'] is False and record['chosen'] == ensemble.REFERENCE_5D
    with pytest.raises(RuntimeError):                                  # 测试集被锁住 / test stays locked
        ensemble.evaluate_test_5d()


def test_5d_rule_allows_test_when_sonnet_clearly_helps(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble, 'SELECTION_5D_PATH', tmp_path / 'sel.json')
    record = ensemble.select_5d(_val_for_5d(sonnet_signal=1.5))      # 更强、独立的信号 / stronger, independent
    assert record['gain_over_reference'] >= ensemble.MIN_GAIN_5D
    assert record['run_test'] is True and record['chosen'] == record['best_with_sonnet']


def test_speaker_prior_uses_train_labels_only(tmp_path, monkeypatch):
    # 改变 val / test 的标签不能改变基线 / changing val / test labels must not change the prior
    from src import external
    ids = [f'l{i}' for i in range(8)]
    meta = pd.DataFrame({'doc_id': ids, 'speaker': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'c']})
    split = ['train'] * 4 + ['val', 'val', 'test', 'test']
    monkeypatch.setattr(external, 'load_liar_metadata', lambda: meta)
    monkeypatch.setattr(ensemble, 'LLM_DIR', tmp_path)
    outs = []
    for later_labels in ([0, 0, 0, 0], [1, 1, 1, 1]):
        liar = pd.DataFrame({'doc_id': ids, 'label': [1, 1, 0, 0] + later_labels, 'split': split})
        monkeypatch.setattr(ensemble, 'load_liar', lambda liar=liar: liar)
        outs.append(ensemble.build_speaker_prior(m=2))
    assert np.allclose(outs[0]['prob::LIAR test'], outs[1]['prob::LIAR test'])
    # a 在训练集里全真，b 全假，c 没出现过 → 总体比例 0.5 / a all true, b all false, c unseen -> 0.5
    p = dict(zip(outs[0]['doc_id::LIAR test'], outs[0]['prob::LIAR test']))
    assert p['l6'] == pytest.approx((2 + 2 * 0.5) / (2 + 2)) and p['l7'] == 0.5


def test_extension_rule_skips_test_when_nothing_passes(tmp_path, monkeypatch):
    val = _val_for_5d(sonnet_signal=1.5)
    val['haiku_context'] = val['haiku']                      # 背景没有带来任何新信息 / context adds nothing
    val['speaker_prior'] = np.random.default_rng(3).uniform(0, 1, len(val))
    monkeypatch.setattr(ensemble, 'LLM_DIR', tmp_path)
    record = ensemble.select_extension('5f', val)
    assert all(not q['run_test'] for q in record['questions'].values())
    out = ensemble.evaluate_extension('5f')                  # 什么也不评估，只记录原因 / nothing evaluated
    assert all(not q['evaluated'] for q in out['questions'].values())
