"""
错误分析规则的单元测试 / Tests for the error-analysis rules.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.error_analysis import agreement_quadrants, bias_by_group, statement_features  # noqa: E402


def test_statement_features():
    text = pd.Series(['Unemployment doubled under his watch.',
                      'Says Obama never visited Israel as president.',
                      'The sky is blue.',
                      'Taxes rose 3 percent.'])
    f = statement_features(text)
    assert f['has number / quantity'].tolist() == [True, False, False, True]   # "doubled"、"3" / "doubled", "3"
    assert f['absolute claim'].tolist() == [False, True, False, False]           # "never"
    assert f['context-dependent reference'].tolist() == [True, False, False, False]  # "his"
    assert f['starts with "Says"'].tolist() == [False, True, False, False]
    # "the" 和 "this" 本身不算指代，只有"这项法案"之类才算 / bare "the"/"this" is not a reference
    assert not statement_features(pd.Series(['This is the best state.']))['context-dependent reference'][0]


def test_agreement_quadrants():
    y = pd.Series([1, 1, 0, 0])
    a = pd.Series([0.9, 0.9, 0.9, 0.1])     # 对、对、错、对 / right, right, wrong, right
    b = pd.Series([0.9, 0.1, 0.9, 0.9])     # 对、错、错、错 / right, wrong, wrong, wrong
    q = agreement_quadrants(y, a, b, 'A', 'B')
    assert q.tolist() == ['both right', 'only A right', 'both wrong', 'only A right']


def test_bias_by_group_sign():
    # 对 g1 偏宽松（全部判真，实际一半为真）/ too lenient on g1: all predicted real, half actually real
    y = pd.Series([1, 0, 1, 0])
    p = pd.Series([0.9, 0.9, 0.9, 0.1])
    g = pd.Series(['g1', 'g1', 'g2', 'g2'])
    out = bias_by_group(y, {'m': p}, g, min_n=1)
    assert out.loc['g1', 'm: predicted - actual'] == 50.0
    assert out.loc['g2', 'm: predicted - actual'] == 0.0
