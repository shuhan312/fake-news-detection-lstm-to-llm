"""
第 4 步方法的单元测试 / Unit tests for the step-4 methods (src/step4.py).

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import step4  # noqa: E402


def test_content_and_style_views_are_exact_complements():
    # 两个视角合起来必须正好是原文的全部词：不能漏词，也不能有词同时出现在两边
    # Together the two views must contain every original word exactly once
    text = pd.Series(['mr smith said on wednesday that the senate bill would cut taxes just here'])
    content = step4.content_only(text)[0].split()
    style = step4.style_only(text)[0].split()
    assert sorted(content + style) == sorted(text[0].split())
    # "bill" 是内容词：sklearn 的停用词表把它当虚词，step4 必须把它保留下来
    # "bill" is a content word: sklearn's list treats it as a stop word; step4 must keep it
    assert content == ['smith', 'senate', 'bill', 'cut', 'taxes']
    assert {'mr', 'said', 'wednesday', 'just', 'here'} <= set(style)


def test_bias_weights_mean_one_and_downweight_easy_articles():
    # 构造数据：真新闻全是"said"文风，假新闻全是"you"文风 → 偏置模型能轻松判对 → 权重应低于平均；
    # 再加几篇"反文风"的文章（真新闻却用 "you"），偏置模型会判错 → 权重应高于平均
    # Easy articles (style matches label) must get below-average weight; "anti-style"
    # articles (real news written with blog words) must get above-average weight
    easy_real = ['said told reporters wednesday officials ' + f'topic{i}' for i in range(40)]
    easy_fake = ['you just this even here ' + f'topic{i}' for i in range(40)]
    hard_real = ['you just this even here ' + f'other{i}' for i in range(5)]
    train = pd.DataFrame({'model_text': easy_real + easy_fake + hard_real,
                          'label': [1] * 40 + [0] * 40 + [1] * 5})
    weights, p_true = step4.bias_weights(train, folds=5)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[:80].mean() < 1 < weights[80:].mean()


def test_leave_outlet_out_never_trains_on_the_held_out_outlet():
    isot = pd.DataFrame({'doc_id': ['i1', 'i2'], 'label': [1, 0], 'model_text': ['a', 'b']})
    welfake = pd.DataFrame({
        'doc_id': [f'w{i}' for i in range(6)],
        'label': [1, 1, 0, 1, 1, 0],
        'split': ['train', 'train', 'train', 'test', 'test', 'test'],
        'model_text': list('cdefgh'),
    })
    groups = pd.Series(['REAL: The New York Times', 'REAL: Breitbart', 'FAKE articles (all)',
                        'REAL: The New York Times', 'REAL: Breitbart', 'FAKE articles (all)'],
                       index=welfake['doc_id'])
    data = step4.leave_outlet_out((isot, isot.iloc[:0]), welfake, groups, 'The New York Times')
    assert 'w0' not in set(data['train']['doc_id'])                 # 留出的出版方不在训练集 / not in train
    assert set(data['train']['doc_id']) == {'i1', 'i2', 'w1', 'w2'}
    assert set(data['test']['doc_id']) == {'w3', 'w5'}              # 只有留出的真新闻 + 假新闻 / held-out real + fake
