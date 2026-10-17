"""
记忆检查：模型是不是"见过"这些陈述？/ Memorisation check: has the model seen these statements?

为什么需要 / Why
    LIAR 来自 2007–2016 年的 PolitiFact 网页，这些网页很可能出现在大模型的训练数据里。
    5d 中 Sonnet 单独就从 Haiku 的 AUC 0.67 跳到 0.77。这可能是因为它知道更多 *事实*，
    也可能是因为它 *记得* PolitiFact 的结论。两者的意义完全不同：后者在新的说法上不会成立。
    LIAR comes from public PolitiFact pages of 2007-2016, likely in the training
    data. Sonnet's jump (AUC 0.67 -> 0.77) could be more knowledge of facts or
    memory of the verdicts; only the former would carry over to new claims.

方法：说话人回忆测试 / Method: speaker recall
    只给模型陈述本身，问"这句话是谁说的"。只测试说话人的名字 *没有出现在陈述里* 的陈述，
    所以不能从文字里直接读出来。如果模型能说出一个地方小政客的名字，它很可能见过这句原话
    （例如在 PolitiFact 的页面上）。这不能直接证明它记得 *评级*，但能衡量它"见过"多少。
    The model sees only the statement and is asked who said it, on statements
    whose speaker's name does not appear in the text. Naming an obscure local
    politician from an exact quote suggests the model has seen it. This measures
    exposure, not recall of the rating itself.

    之后在 notebook 中比较：每个模型在"认出说话人"和"没认出"的陈述上的 AUC。如果 Sonnet 比
    Haiku 的优势集中在"认出"的陈述上，说明它的提升有一部分来自记忆。
    The notebook then compares each model's AUC on recalled vs not-recalled statements.

只在 test 上运行：这是对已经固定的模型的描述，不用来选择或调整任何东西。
Run on test only: it describes fixed models and selects nothing.

命令行 / Command line:
    python -m src.memorisation --submit haiku     # 或 sonnet / or sonnet
    python -m src.memorisation --collect haiku
"""

import argparse
import json
import re
import time

import pandas as pd

from src.data import PROJECT_ROOT
from src.external import load_liar, load_liar_metadata
from src.llm_factcheck import MODELS, SONNET_THINKING

RESULTS_DIR = PROJECT_ROOT / 'results' / 'llm'

PROBE_SYSTEM = (
    'You will be given a statement from US politics. Who made this statement? If you are not sure, '
    'give your best guess. Reply with the name of the person or organisation only.')
PROBE_FORMAT = {'format': {'type': 'json_schema', 'schema': {
    'type': 'object', 'properties': {'speaker': {'type': 'string'}},
    'required': ['speaker'], 'additionalProperties': False}}}

# 只测试"个人"说话人：chain-email、facebook-posts 之类的来源没有可以回忆的名字
# Only person-like speakers: sources such as chain-email have no name to recall
PERSON_PARTIES = {'democrat', 'republican', 'independent', 'libertarian', 'green', 'constitution-party',
                  'tea-party-member', 'democratic-farmer-labor'}


def _tokens(text: str) -> set:
    return set(re.findall(r'[a-z]+', str(text).lower()))


def probe_set(split: str = 'test') -> pd.DataFrame:
    """可以测试的陈述：说话人是个人，而且名字（任何一部分）都没有出现在陈述里。
    Testable statements: a person speaker whose name (any part of it) does not appear in the text."""
    liar = load_liar().set_index('doc_id')
    meta = load_liar_metadata().set_index('doc_id')
    df = liar.loc[liar['split'] == split, ['text', 'label', 'label6']].join(meta[['speaker', 'party']])
    name_parts = df['speaker'].str.split('-')
    named = [bool(set(parts) & _tokens(t)) for parts, t in zip(name_parts, df['text'])]
    keep = df['party'].isin(PERSON_PARTIES) & ~pd.Series(named, index=df.index)
    return df[keep]


def is_correct(answer: str, speaker_slug: str) -> bool:
    """严格匹配：说话人名字的每一部分都要出现在回答里（"hillary-clinton" 需要 hillary 和 clinton）。
    只答姓不算对，避免 "Clinton" 同时匹配比尔和希拉里、"Paul" 同时匹配兰德和罗恩。
    Strict: every part of the speaker's name must appear in the answer; a bare
    surname does not count (Clinton could be Bill or Hillary)."""
    return set(speaker_slug.split('-')) <= _tokens(answer)


def parse_speaker(text: str) -> str:
    try:
        value = json.loads(text).get('speaker', '')
    except (TypeError, ValueError, AttributeError):
        return ''
    return value if isinstance(value, str) else ''


def _params(statement: str, model: str) -> dict:
    params = {'model': MODELS[model], 'max_tokens': 30, 'temperature': 0.0, 'system': PROBE_SYSTEM,
              'output_config': PROBE_FORMAT,
              'messages': [{'role': 'user', 'content': f'Statement: {statement}'}]}
    if model == 'sonnet':            # 和 5d 相同的设置 / same settings as 5d
        del params['temperature']
        params['thinking'] = SONNET_THINKING
        params['max_tokens'] = 40
    return params


def submit(model: str) -> str:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    df = probe_set()
    batch = anthropic.Anthropic().messages.batches.create(requests=[
        Request(custom_id=d, params=MessageCreateParamsNonStreaming(**_params(t, model)))
        for d, t in zip(df.index, df['text'])])
    (RESULTS_DIR / f'probe_speaker_{model}_batch.json').write_text(json.dumps(
        {'batch_id': batch.id, 'model': MODELS[model], 'n_requests': len(df), 'system_prompt': PROBE_SYSTEM}, indent=2))
    print(f'submitted probe batch {batch.id} with {len(df)} requests')
    return batch.id


def retry_failed(model: str) -> str:
    """只重新提交失败的请求（例如余额不足时被拒绝的），和 src/llm_factcheck.retry_failed 相同的做法。
    Resubmit only the failed requests, as in src/llm_factcheck.retry_failed."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    path = RESULTS_DIR / f'probe_speaker_{model}_batch.json'
    meta = json.loads(path.read_text())
    bids = [meta['batch_id']] + meta.get('retry_batch_ids', [])
    done = set()
    for bid in bids:
        if client.messages.batches.retrieve(bid).processing_status != 'ended':
            raise RuntimeError(f'batch {bid} is still running')
        done |= {r.custom_id for r in client.messages.batches.results(bid) if r.result.type == 'succeeded'}
    df = probe_set()
    todo = df[~df.index.isin(done)]
    if todo.empty:
        print('nothing to retry')
        return None
    batch = client.messages.batches.create(requests=[
        Request(custom_id=d, params=MessageCreateParamsNonStreaming(**_params(t, model)))
        for d, t in zip(todo.index, todo['text'])])
    meta['retry_batch_ids'] = meta.get('retry_batch_ids', []) + [batch.id]
    path.write_text(json.dumps(meta, indent=2))
    print(f'resubmitted {len(todo)} failed probe requests as batch {batch.id}')
    return batch.id


def collect(model: str, poll_seconds: int = 60) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    meta = json.loads((RESULTS_DIR / f'probe_speaker_{model}_batch.json').read_text())
    bids = [meta['batch_id']] + meta.get('retry_batch_ids', [])
    for bid in bids:
        while client.messages.batches.retrieve(bid).processing_status != 'ended':
            time.sleep(poll_seconds)
    # 原始批次和重试批次合并：任何一次成功的回答都会被采用 / merge the original and retry batches
    answers, usage = {}, {'input_tokens': 0, 'output_tokens': 0}
    for bid in bids:
        for result in client.messages.batches.results(bid):
            if result.result.type == 'succeeded':
                msg = result.result.message
                answers[result.custom_id] = parse_speaker(next((b.text for b in msg.content if b.type == 'text'), ''))
                usage['input_tokens'] += msg.usage.input_tokens
                usage['output_tokens'] += msg.usage.output_tokens
    df = probe_set()
    missing = [d for d in df.index if d not in answers]
    if missing:
        # 没有回答的请求不能当作"答错"，否则会低估回忆率 / unanswered must not count as wrong
        raise RuntimeError(f'{len(missing)} probe requests never succeeded; run --retry {model} first')
    rows = {d: {'answer': answers.get(d, ''), 'speaker': s, 'correct': is_correct(answers.get(d, ''), s)}
            for d, s in zip(df.index, df['speaker'])}
    record = {**meta, 'usage': usage, 'n_answered': len(answers),
              'recall %': round(100 * sum(r['correct'] for r in rows.values()) / len(rows), 1), 'rows': rows}
    (RESULTS_DIR / f'probe_speaker_{model}.json').write_text(json.dumps(record, indent=2))
    print(f"{model}: recall {record['recall %']}% of {len(rows)} | tokens {usage}")
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speaker-recall memorisation check on LIAR test.')
    parser.add_argument('--submit', choices=list(MODELS))
    parser.add_argument('--collect', choices=list(MODELS))
    parser.add_argument('--retry', choices=list(MODELS))
    args = parser.parse_args()
    if args.submit:
        submit(args.submit)
    if args.retry:
        retry_failed(args.retry)
    if args.collect:
        collect(args.collect)
