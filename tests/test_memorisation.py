"""
记忆检查规则的单元测试 / Tests for the memorisation-check rules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.memorisation import is_correct, parse_speaker, probe_set  # noqa: E402


def test_is_correct_requires_the_full_name():
    assert is_correct('Hillary Clinton', 'hillary-clinton')
    assert is_correct('Senator John McCain (R-AZ)', 'john-mccain')
    assert not is_correct('Clinton', 'hillary-clinton')          # 只答姓不算 / surname alone is not enough
    assert not is_correct('Bill Clinton', 'hillary-clinton')
    assert not is_correct('', 'rick-scott')


def test_parse_speaker():
    assert parse_speaker('{"speaker": "Rick Scott"}') == 'Rick Scott'
    assert parse_speaker('Rick Scott') == '' and parse_speaker(None) == ''


def test_probe_set_excludes_names_in_the_text_and_non_persons():
    df = probe_set('test')
    assert len(df) > 300
    assert set(df['party']) & {'none', 'organization'} == set()
    for slug, text in zip(df['speaker'], df['text']):
        assert not any(part in text.lower().split() for part in slug.split('-'))
