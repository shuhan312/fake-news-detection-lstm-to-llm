"""
第 6 步（WELFake 跨来源测试）的单元测试，不调用 API / Tests for step 6 (no API calls).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import llm_crosssource as X  # noqa: E402


def test_article_input_truncates_and_marks_the_cut():
    text = ' '.join(f'w{i}' for i in range(500))
    out = X.article_input('A title', text, max_words=300)
    assert out.startswith('Title: A title\n\nArticle: w0 w1')
    assert out.endswith('w299 ...') and 'w300' not in out
    assert not X.article_input('T', 'short body').endswith('...')      # 没截断就不加 / no cut, no marker


def test_params_match_the_liar_settings():
    h = X._params('T', 'body', 'haiku')
    s = X._params('T', 'body', 'sonnet')
    assert h['temperature'] == 0.0 and 'thinking' not in h
    assert 'temperature' not in s and s['thinking'] == {'type': 'between_tools'}
    assert h['system'] == s['system'] == X.SYSTEM_ARTICLE
    assert 'style' not in X.SYSTEM_ARTICLE.lower()                      # 提示词不提文风 / no style hint

