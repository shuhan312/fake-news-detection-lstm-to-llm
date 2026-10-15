"""
实验运行器和模型配置的单元测试（不训练模型）/ Tests for the runner and model configs (no training).

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import experiments, lstm_model  # noqa: E402


def test_ablation_ladder_changes_exactly_one_setting_per_step():
    # 消融实验的前提：每一步只比上一步多改一项设置（name / description 除外）
    # The point of the ablation: each step changes exactly ONE setting
    ladder = [lstm_model.CONFIGS[n] for n in lstm_model.LADDER]
    for prev, cur in zip(ladder, ladder[1:]):
        a, b = lstm_model.config_dict(prev), lstm_model.config_dict(cur)
        changed = [k for k in a if k not in ('name', 'description') and a[k] != b[k]]
        assert len(changed) == 1, f'{cur.name} changes {changed}'


def test_a0_matches_the_original_paper():
    a0 = lstm_model.CONFIGS['A0']
    assert (a0.vocab_size, a0.maxlen, a0.embedding_dim, a0.lstm_units,
            a0.bidirectional, a0.dense_dropout, a0.oov_token) == (5000, 200, 100, 128, False, 0.5, False)


def test_summarize_reports_mean_std_and_seed_count():
    results = pd.DataFrame({
        'model': ['lstm'] * 3, 'config': ['A0'] * 3, 'eval set': ['ISOT test'] * 3,
        'accuracy %': [97.0, 98.0, 99.0], 'balanced accuracy %': [97.0, 98.0, 99.0],
        'AUC': [0.990, 0.995, 1.000],
    })
    row = experiments.summarize(results).loc[('lstm', 'A0', 'ISOT test')]
    assert row['accuracy %'] == '98.0 ± 1.0'
    assert row['seeds'] == 3


def test_new_settings_keep_a0_to_a5_unchanged():
    # A6 新增的设置，默认值必须等于原论文 / Keras 默认，否则已经跑完的 A0–A5 结果就和代码对不上了
    # A6's new settings must default to the paper / Keras values, or the finished
    # A0–A5 results would no longer match the code
    for name in lstm_model.LADDER:
        c = lstm_model.CONFIGS[name]
        assert (c.padding, c.mask_zero, c.learning_rate, c.clipnorm) == ('pre', False, 1e-3, None)
    assert 'A6' in lstm_model.CONFIGS and 'A6' not in lstm_model.LADDER
