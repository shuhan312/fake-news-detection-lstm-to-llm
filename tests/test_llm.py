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


def test_run_name_keeps_test_files_and_separates_val():
    # 验证集的结果不能覆盖已经保存的测试集结果 / val results must never overwrite the saved test results
    from src.llm_factcheck import run_name
    assert run_name('no_evidence') == 'no_evidence'
    assert run_name('no_evidence', 'test') == 'no_evidence'
    assert run_name('no_evidence', 'val') == 'no_evidence_val'
    import pytest
    with pytest.raises(ValueError):
        run_name('no_evidence', 'train')


def test_collect_saves_val_results_under_the_run_name(tmp_path, monkeypatch):
    # 曾经的错误：循环变量覆盖了文件名，结果被存成 "llm_LIAR val" / a loop variable once renamed the file
    import json as _json
    from types import SimpleNamespace as NS
    import pandas as pd
    from src import llm_factcheck as L

    df = pd.DataFrame({'doc_id': ['v1', 'v2'], 'text': ['a', 'b'], 'label': [1, 0]})
    monkeypatch.setattr(L, '_liar_sets', lambda: {'LIAR val': df, 'LIAR test': df})
    monkeypatch.setattr(L, 'RESULTS_DIR', tmp_path)
    (tmp_path / 'no_evidence_val_batch.json').write_text(_json.dumps({'batch_id': 'b', 'sets': ['LIAR val']}))

    def result(doc_id, p):
        msg = NS(content=[NS(type='text', text=f'{{"probability": {p}}}')], usage=NS(input_tokens=1, output_tokens=1))
        return NS(custom_id=doc_id, result=NS(type='succeeded', message=msg))
    batches = NS(retrieve=lambda _id: NS(processing_status='ended'),
                 results=lambda _id: [result('v1', 80), result('v2', 20)])
    monkeypatch.setattr(L, '_client', lambda: NS(messages=NS(batches=batches)))

    L.collect('no_evidence', 'val', poll_seconds=0)
    assert sorted(p.name for p in tmp_path.glob('llm_*')) == ['llm_no_evidence_val.json', 'llm_no_evidence_val.npz']
    assert _json.loads((tmp_path / 'llm_no_evidence_val.json').read_text())['metrics']['LIAR val']['AUC'] == 1.0


def test_haiku_requests_are_unchanged_and_sonnet_differs_only_where_required():
    from src import llm_factcheck as L
    h = L._params('Some statement.')
    # Haiku：和 5a–5c 实际发送的请求完全一样 / Haiku: exactly what 5a-5c sent
    assert h['model'] == 'claude-haiku-4-5' and h['temperature'] == 0.0 and h['max_tokens'] == 20
    assert 'thinking' not in h
    s = L._params('Some statement.', model='sonnet')
    # Sonnet：同样的提示词、输出格式和输入；不传 temperature，关掉扩展思考
    # Sonnet: same prompt, format and input; no temperature, extended thinking off
    assert s['model'] == 'claude-sonnet-5-5'
    assert 'temperature' not in s and s['thinking'] == {'type': 'between_tools'}
    assert {k: s[k] for k in ('system', 'output_config', 'messages')} == \
           {k: h[k] for k in ('system', 'output_config', 'messages')}


def test_run_name_separates_models():
    from src.llm_factcheck import run_name
    assert run_name('no_evidence', 'test', 'haiku') == 'no_evidence'          # 原有文件名不变 / unchanged
    assert run_name('no_evidence', 'val', 'sonnet') == 'no_evidence_sonnet_val'
    assert run_name('no_evidence', 'test', 'sonnet') == 'no_evidence_sonnet'


def test_format_context_names_speaker_and_omits_empty_fields():
    from src.llm_factcheck import format_context
    full = {'speaker': 'rick-scott', 'job_title': 'Governor', 'state': 'Florida', 'context': 'a campaign ad'}
    assert format_context(full) == 'Speaker: Rick Scott (Governor, Florida)\nSetting: a campaign ad'
    bare = {'speaker': 'chain-email', 'job_title': '', 'state': ' ', 'context': ''}
    assert format_context(bare) == 'Speaker: Chain Email'          # 空字段不写 "unknown" / no "unknown"
    assert 'party' not in format_context({**full, 'party': 'republican'}).lower()   # 不给党派 / no party


def test_context_params_differ_from_5a_only_in_prompt_and_context():
    from src import llm_factcheck as L
    a = L._params('Taxes rose.')
    f = L._params('Taxes rose.', context='Speaker: Rick Scott')
    assert f['messages'][0]['content'] == 'Speaker: Rick Scott\nStatement: Taxes rose.'
    assert f['system'] == L.SYSTEM_CONTEXT and f['system'] != a['system']
    assert {k: f[k] for k in ('model', 'max_tokens', 'temperature', 'output_config')} == \
           {k: a[k] for k in ('model', 'max_tokens', 'temperature', 'output_config')}
    import pytest
    with pytest.raises(ValueError):
        L._params('x', passages=[], context='y')


def test_shuffled_context_never_keeps_its_own(monkeypatch):
    import pandas as pd
    from src import external, llm_factcheck as L
    ids = [f'liar_{i}' for i in range(40)]
    meta = pd.DataFrame({'doc_id': ids, 'speaker': [f'person-{i}' for i in range(40)],
                         'job_title': '', 'state': '', 'context': ''})
    monkeypatch.setattr(external, 'load_liar_metadata', lambda: meta)
    own, shuffled = L.contexts_for('context', ids), L.contexts_for('shuffled_context', ids)
    assert all(own[d] == f'Speaker: Person {d.split("_")[1]}' for d in ids)
    assert all(shuffled[d] != own[d] for d in ids)
    assert sorted(shuffled.values()) == sorted(own.values())     # 每份背景恰好用一次 / each used once


def test_collect_merges_a_retry_batch(tmp_path, monkeypatch):
    # 原始批次里 v2 因余额不足失败，重试批次里成功：最终不应有失败，也不应有 0.5
    # v2 failed in the original batch and succeeded in the retry: no failures, no 0.5 fill-in
    import json as _json
    from types import SimpleNamespace as NS
    import numpy as np
    import pandas as pd
    from src import llm_factcheck as L

    df = pd.DataFrame({'doc_id': ['v1', 'v2'], 'text': ['a', 'b'], 'label': [1, 0]})
    monkeypatch.setattr(L, '_liar_sets', lambda: {'LIAR val': df, 'LIAR test': df})
    monkeypatch.setattr(L, 'RESULTS_DIR', tmp_path)
    (tmp_path / 'no_evidence_val_batch.json').write_text(_json.dumps(
        {'batch_id': 'b1', 'retry_batch_ids': ['b2'], 'sets': ['LIAR val']}))

    def ok(doc_id, p):
        msg = NS(content=[NS(type='text', text=f'{{"probability": {p}}}')], usage=NS(input_tokens=1, output_tokens=1))
        return NS(custom_id=doc_id, result=NS(type='succeeded', message=msg))
    err = lambda doc_id: NS(custom_id=doc_id, result=NS(type='errored'))
    results = {'b1': [ok('v1', 80), err('v2')], 'b2': [ok('v2', 20)]}
    batches = NS(retrieve=lambda _id: NS(processing_status='ended'), results=lambda bid: results[bid])
    monkeypatch.setattr(L, '_client', lambda: NS(messages=NS(batches=batches)))

    L.collect('no_evidence', 'val', poll_seconds=0)
    rec = _json.loads((tmp_path / 'llm_no_evidence_val.json').read_text())
    assert rec['unparsed_or_failed']['LIAR val'] == 0 and rec['failed_types'] == {}
    z = np.load(tmp_path / 'llm_no_evidence_val.npz')
    assert list(z['prob::LIAR val']) == [0.8, 0.2]
