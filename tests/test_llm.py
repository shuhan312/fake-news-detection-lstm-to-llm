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
