"""
LLM 回答解析的单元测试（不调用 API）/ Tests for parsing LLM replies (no API calls).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm_factcheck import parse_probability  # noqa: E402


def test_parse_probability_reads_structured_json():
    assert parse_probability('{"probability": 85}') == 0.85
    assert parse_probability('{"probability": 100}') == 1.0
    assert parse_probability('{"probability": 0}') == 0.0
    assert parse_probability('{"probability": 130}') == 1.0      # 截断到 100 / clipped


def test_parse_probability_rejects_free_text():
    # 不能从推理文字里抓数字：小样本测试中，正则把陈述里的 "66 of the first 100 days" 误读为 66%
    # Never grab numbers from free text: regex read "66 of the first 100 days" as 66%
    assert math.isnan(parse_probability('I need to evaluate whether Congress spent 66 of the first 100 days'))
    assert math.isnan(parse_probability('85'))
    assert math.isnan(parse_probability(''))
    assert math.isnan(parse_probability(None))
    assert math.isnan(parse_probability('{"probability": "high"}'))


def test_parse_relevant_and_probability_from_evidence_reply():
    from src.llm_factcheck import parse_relevant
    reply = '{"probability": 20, "evidence_relevant": true}'
    assert parse_probability(reply) == 0.20
    assert parse_relevant(reply) is True
    assert parse_relevant('{"probability": 20}') is None
    assert parse_relevant('not json') is None


def test_random_evidence_control_never_gives_a_statement_its_own_evidence(monkeypatch):
    from src import evidence, llm_factcheck
    ids = [f'liar_{i}' for i in range(50)]
    fake = {d: {'passages': [{'title': d, 'text': f'evidence for {d}'}]} for d in ids}
    monkeypatch.setattr(evidence, 'load_evidence', lambda split='test': fake)
    own = llm_factcheck.passages_for('evidence', ids)
    shuffled = llm_factcheck.passages_for('random_evidence', ids)
    assert all(own[d][0]['title'] == d for d in ids)
    assert all(shuffled[d][0]['title'] != d for d in ids)                  # 没有一条拿到自己的证据 / no self-matches
    assert sorted(p[0]['title'] for p in shuffled.values()) == sorted(ids)  # 每份证据恰好用一次 / each used once
