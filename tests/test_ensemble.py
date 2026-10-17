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
