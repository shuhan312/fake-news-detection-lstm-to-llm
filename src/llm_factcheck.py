"""
第 5 步：用大语言模型判断 LIAR 陈述的真假 / Step 5: LLM fact-checking on LIAR.

为什么用 LLM / Why an LLM
    07_liar_benchmark 表明：去掉来源捷径后，只看文字的模型（TF-IDF、DistilBERT）在 LIAR 上
    只有 AUC 约 0.68。一句话是真是假往往取决于外部事实，而这些模型只看得到这句话本身。
    大语言模型带着大量世界知识，并且能阅读证据、做推理。
    Text-only models top out around AUC 0.68 on LIAR because truth depends on
    outside facts; an LLM brings world knowledge and can reason over evidence.

成本 / Cost
    使用最便宜的 Claude Haiku 4.5（输入 $1、输出 $5 每百万 token），并通过 Batch API
    提交（不要求即时返回，价格五折）。每条请求只有一两百个 token，输出只有一个数字。
    Cheapest model (Claude Haiku 4.5, $1 / $5 per MTok) via the Batch API (50% off);
    each request is a couple of hundred tokens and the answer is a single number.

实验 / Experiments
    5a  no_evidence   只给陈述，让模型凭自身知识判断。这也是"记忆"检查：LIAR 来自 2007–2016 年
                      的 PolitiFact，核查结论是公开网页，可能出现在模型的训练数据里。
                      Statement only. Also a memorisation check: LIAR's PolitiFact
                      verdicts are public web pages the model may have seen.

    5b  evidence      陈述 + 为它检索到的维基百科证据（src/evidence.py）
                      statement + Wikipedia evidence retrieved for it
        random_evidence  对照组：陈述 + *另一条* 陈述的证据。格式和长度一样，只是内容不相关。
                      如果 5b 的提升只来自"提示变长、格式变了"，这个对照组也会一样好。
                      control: statement + ANOTHER statement's evidence (same format,
                      unrelated content). If 5b's gain came only from the longer prompt,
                      this control would match it.

只使用陈述文本，和 07 中的模型保持一致（不给说话人、党派等信息）。
Only the statement text is given, as for the models in notebook 07.

命令行 / Command line:
    python -m src.llm_factcheck --sample 20          # 小样本即时测试，检查输出格式 / quick sync check
    python -m src.llm_factcheck --submit no_evidence # 提交批量任务 / submit the batch
    python -m src.llm_factcheck --collect no_evidence  # 等待并收集结果 / wait and collect
"""

import argparse
import json
import time

import numpy as np

from src import baselines as B
from src.data import PROJECT_ROOT
from src.external import load_liar

MODEL = 'claude-haiku-4-5'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'llm'

# 提示词：要求模型只输出一个 0–100 的整数，表示"属实或基本属实"的概率。
# 用概率而不是"真/假"，是为了能计算 AUC，和 TF-IDF、DistilBERT 公平比较。
# The model returns one integer 0-100: the probability the statement is true or
# mostly true. A probability (not a label) gives an AUC comparable with notebook 07.
SYSTEM_NO_EVIDENCE = (
    'You are a professional political fact-checker. You will be given a statement made by a '
    'US political figure. Estimate the probability that a professional fact-checker would rate '
    'the statement as true, mostly true, or half true (rather than barely true, false, or '
    '"pants on fire"). Use your general knowledge. Reply with only an integer from 0 to 100.')

# 结构化输出：强制模型只能返回 {"probability": 整数}。
# 第一次小样本测试时只用文字要求"只回答一个整数"，20 条里有 16 条没有遵守，模型开始写推理并被截断；
# 而且从推理文字里用正则取数字会出错（例如把陈述中的 "66 of the first 100 days" 当成概率）。
# 用 output_config.format 从机制上保证格式，同时输出最短、成本最低。
# Structured output forces {"probability": int}. In the first sample, a text-only
# instruction was ignored 16 times out of 20 (the model started reasoning and was
# cut off), and regex-parsing that reasoning grabbed wrong numbers (e.g. "66 of
# the first 100 days"). output_config.format guarantees the format and keeps the
# output - and the cost - minimal.
OUTPUT_FORMAT = {
    'format': {
        'type': 'json_schema',
        'schema': {
            'type': 'object',
            'properties': {'probability': {'type': 'integer'}},
            'required': ['probability'],
            'additionalProperties': False,
        },
    },
}

# JSON 只有十几个字符，20 个 token 足够；temperature=0 让同一条陈述每次得到相同的回答
# The JSON is ~15 characters, so 20 tokens suffice; temperature 0 makes answers repeatable
MAX_TOKENS = 20
TEMPERATURE = 0.0

# 5b 的提示词：和 5a 相同的任务，只是多了证据；明确要求"证据不相关时依靠常识"，
# 这样证据没用的陈述不会因为"被迫使用证据"而判断得更差
# 5b prompt: the same task as 5a plus evidence; told to fall back on general
# knowledge when the evidence is irrelevant, so useless evidence cannot force worse answers
SYSTEM_EVIDENCE = (
    'You are a professional political fact-checker. You will be given a statement made by a '
    'US political figure, followed by passages retrieved from Wikipedia. Estimate the probability '
    'that a professional fact-checker would rate the statement as true, mostly true, or half true '
    '(rather than barely true, false, or "pants on fire"). Use the passages when they are relevant; '
    'when they are not, rely on your general knowledge. Also say whether the passages were '
    'relevant to checking the statement.')

# 多一个字段 evidence_relevant：模型认为证据是否有用，之后可以按它分组分析
# Extra field evidence_relevant: whether the model found the passages useful, for analysis
OUTPUT_FORMAT_EVIDENCE = {
    'format': {
        'type': 'json_schema',
        'schema': {
            'type': 'object',
            'properties': {'probability': {'type': 'integer'},
                           'evidence_relevant': {'type': 'boolean'}},
            'required': ['probability', 'evidence_relevant'],
            'additionalProperties': False,
        },
    },
}
MAX_TOKENS_EVIDENCE = 40     # 两个字段的 JSON 约 25 个 token / a two-field JSON is ~25 tokens


def _client():
    import anthropic
    return anthropic.Anthropic()


def format_evidence(passages: list) -> str:
    """把证据段落排成带编号和文章标题的文本 / Number the passages with their article titles."""
    if not passages:
        return 'Wikipedia passages: none were found.'
    return 'Wikipedia passages:\n' + '\n'.join(
        f'[{i}] ({p["title"]}) {p["text"]}' for i, p in enumerate(passages, 1))


def _params(statement: str, passages: list = None) -> dict:
    """一条陈述对应的请求参数；passages 为 None 时是 5a（不给证据），否则是 5b。
    Request parameters; passages=None gives 5a (no evidence), otherwise 5b."""
    if passages is None:
        return {'model': MODEL, 'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE,
                'system': SYSTEM_NO_EVIDENCE, 'output_config': OUTPUT_FORMAT,
                'messages': [{'role': 'user', 'content': f'Statement: {statement}'}]}
    return {'model': MODEL, 'max_tokens': MAX_TOKENS_EVIDENCE, 'temperature': TEMPERATURE,
            'system': SYSTEM_EVIDENCE, 'output_config': OUTPUT_FORMAT_EVIDENCE,
            'messages': [{'role': 'user',
                          'content': f'Statement: {statement}\n\n{format_evidence(passages)}'}]}


def parse_relevant(text: str):
    """读出 evidence_relevant（True / False），没有这个字段返回 None。
    Read evidence_relevant (True / False); None if absent."""
    try:
        value = json.loads(text).get('evidence_relevant')
    except (TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def passages_for(experiment: str, doc_ids, seed: int = 42) -> dict:
    """每条陈述要给模型看的证据：5a 为 None；5b 是它自己的证据；对照组是另一条陈述的证据。
    Evidence shown per statement: None (5a), its own (5b) or another statement's (control).

    对照组用"错位"分配：先固定种子打乱顺序，再整体移动一位，保证没有任何一条拿到自己的证据。
    The control uses a derangement: shuffle with a fixed seed, then shift by one, so
    no statement ever receives its own evidence.
    """
    doc_ids = list(doc_ids)
    if experiment == 'no_evidence':
        return {d: None for d in doc_ids}
    from src.evidence import load_evidence
    evidence = load_evidence('test')
    own = {d: evidence.get(d, {}).get('passages', []) for d in doc_ids}
    if experiment == 'evidence':
        return own
    if experiment == 'random_evidence':
        order = list(np.random.default_rng(seed).permutation(doc_ids))
        donor = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        return {d: own[donor[d]] for d in doc_ids}
    raise ValueError(f'unknown experiment: {experiment}')


def parse_probability(text: str) -> float:
    """从结构化输出 {"probability": 整数} 中读出概率（0–1）；格式不对就返回 NaN。
    Read the probability (0-1) from the structured {"probability": int} reply; NaN otherwise.

    只接受 JSON，*不* 再用正则表达式在文字里找数字：那样会把陈述里出现的数字误当成概率。
    超出 0–100 的值截断到边界。
    Only JSON is accepted - no regex search in free text, which mistook numbers
    quoted from the statement for probabilities. Values are clipped to 0-100.
    """
    try:
        value = json.loads(text)['probability']
    except (TypeError, ValueError, KeyError):
        return float('nan')
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return float('nan')
    return min(max(float(value), 0.0), 100.0) / 100


def _liar_sets():
    liar = load_liar()
    return {name: liar[liar['split'] == split].reset_index(drop=True)
            for name, split in (('LIAR val', 'val'), ('LIAR test', 'test'))}


def run_sample(n: int = 20):
    """即时调用（非批量）测试 n 条验证集陈述：检查回答格式能否解析，并估算 token 用量。
    Synchronous check on n validation statements: parsing and token usage."""
    client = _client()
    val = _liar_sets()['LIAR val'].head(n)
    usage_in = usage_out = 0
    for _, row in val.iterrows():
        msg = client.messages.create(**_params(row['text']))
        reply = next((b.text for b in msg.content if b.type == 'text'), '')
        usage_in += msg.usage.input_tokens
        usage_out += msg.usage.output_tokens
        print(f"{row['label6']:>12} | P(true)={parse_probability(reply)} | reply={reply!r} | {row['text'][:70]}")
    print(f'\n{n} requests: {usage_in} input + {usage_out} output tokens '
          f'({usage_in / n:.0f} / {usage_out / n:.1f} per request)')
    return usage_in / n, usage_out / n


def submit(experiment: str = 'no_evidence', sets=('LIAR test',)) -> str:
    """把指定数据集的全部陈述作为一个批量任务提交，保存批次 ID。
    Submit every statement of the chosen sets as one batch; save the batch id.

    默认只提交 LIAR test（1,283 条）：5a 不需要调参，验证集用不上，只跑测试集可以把费用减半。
    Default is LIAR test only (1,283): 5a has nothing to tune, so skipping val halves the cost.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    chosen = {k: v for k, v in _liar_sets().items() if k in sets}
    requests = []
    for df in chosen.values():
        shown = passages_for(experiment, df['doc_id'])
        requests += [Request(custom_id=doc_id,
                             params=MessageCreateParamsNonStreaming(**_params(text, shown[doc_id])))
                     for doc_id, text in zip(df['doc_id'], df['text'])]
    batch = _client().messages.batches.create(requests=requests)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{experiment}_batch.json').write_text(json.dumps(
        {'batch_id': batch.id, 'model': MODEL, 'n_requests': len(requests), 'sets': list(chosen),
         'system_prompt': SYSTEM_NO_EVIDENCE if experiment == 'no_evidence' else SYSTEM_EVIDENCE,
         'max_tokens': MAX_TOKENS if experiment == 'no_evidence' else MAX_TOKENS_EVIDENCE,
         'temperature': TEMPERATURE}, indent=2))
    print(f'submitted batch {batch.id} with {len(requests)} requests')
    return batch.id


def collect(experiment: str = 'no_evidence', poll_seconds: int = 60):
    """等待批量任务完成，取回结果，计算指标并保存（格式与 results/liar 相同）。
    Wait for the batch, fetch results, compute metrics, save (same format as results/liar)."""
    meta = json.loads((RESULTS_DIR / f'{experiment}_batch.json').read_text())
    client = _client()
    while True:
        batch = client.messages.batches.retrieve(meta['batch_id'])
        if batch.processing_status == 'ended':
            break
        print(f'status {batch.processing_status}: {batch.request_counts.processing} processing', flush=True)
        time.sleep(poll_seconds)

    # 结果的返回顺序是任意的，必须按 custom_id（也就是 doc_id）对应，不能按位置
    # Results arrive in any order: key them by custom_id (= doc_id), never by position
    replies, usage_in, usage_out, failed = {}, 0, 0, {}
    for result in client.messages.batches.results(meta['batch_id']):
        if result.result.type == 'succeeded':
            msg = result.result.message
            replies[result.custom_id] = next((b.text for b in msg.content if b.type == 'text'), '')
            usage_in += msg.usage.input_tokens
            usage_out += msg.usage.output_tokens
        else:
            failed[result.custom_id] = result.result.type

    eval_sets = {k: v for k, v in _liar_sets().items() if k in meta['sets']}
    probs, metrics, unparsed, relevant_share = {}, {}, {}, {}
    for name, df in eval_sets.items():
        relevant = [parse_relevant(replies.get(d)) for d in df['doc_id']]
        flags = [r for r in relevant if r is not None]
        relevant_share[name] = round(100 * sum(flags) / len(flags), 1) if flags else None
        p = np.array([parse_probability(replies.get(d)) for d in df['doc_id']])
        unparsed[name] = int(np.isnan(p).sum())
        # 无法解析或失败的请求按 0.5（"不知道"）处理，并单独报告数量
        # Unparsed / failed answers count as 0.5 ("don't know") and are reported
        probs[name] = np.where(np.isnan(p), 0.5, p)
        metrics[name] = B.evaluate(df['label'], probs[name])

    record = {'name': f'llm_{experiment}', **meta, 'metrics': metrics,
              'unparsed_or_failed': unparsed, 'failed_types': failed,
              'evidence_relevant_%': relevant_share,
              'usage': {'input_tokens': usage_in, 'output_tokens': usage_out}}
    path = RESULTS_DIR / f'llm_{experiment}'
    path.with_suffix('.json').write_text(json.dumps(record, indent=2))
    np.savez_compressed(path.with_suffix('.npz'),
                        **{f'prob::{k}': probs[k] for k in eval_sets},
                        **{f'doc_id::{k}': df['doc_id'].to_numpy(dtype=str) for k, df in eval_sets.items()},
                        **{f'reply::{k}': np.array([replies.get(d, '') for d in df['doc_id']], dtype=str)
                           for k, df in eval_sets.items()})
    print('saved', path.name, {k: (m['accuracy %'], m['AUC']) for k, m in metrics.items()},
          '| unparsed/failed:', unparsed, '| tokens:', record['usage'])


def main():
    parser = argparse.ArgumentParser(description='Step 5: LLM fact-checking on LIAR.')
    parser.add_argument('--sample', type=int, help='synchronous test on N validation statements')
    parser.add_argument('--submit', help='experiment name to submit as a batch')
    parser.add_argument('--collect', help='experiment name to collect')
    args = parser.parse_args()
    if args.sample:
        run_sample(args.sample)
    if args.submit:
        submit(args.submit)
    if args.collect:
        collect(args.collect)


if __name__ == '__main__':
    main()
