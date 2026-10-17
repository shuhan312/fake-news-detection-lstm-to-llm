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

    5f  context       陈述 + 发言背景：说话人、职务、州、场合（例如 "a campaign ad"）。
                      notebook 09 发现：含"他""这项法案"等指代的陈述上，Haiku 接近随机（AUC 0.568）。
                      5f 直接检验"缺少发言背景"是不是瓶颈。
                      statement + who said it, their job and state, and the setting.
                      Notebook 09 found Haiku near chance on statements with
                      references like "he" or "this bill"; 5f tests that bottleneck.
        shuffled_context  对照组：陈述 + *另一条* 陈述的发言背景（和 5b 的对照组同样的错位分配）。
                      区分"背景的内容有用"和"提示格式变了"。
                      control: another statement's context (same derangement as 5b).

    5f 刻意 *不* 给党派：党派标签本身就是"按来源判断"的捷径，而且说话人的名字已经提供了
    消除指代所需的信息。历史核查次数也不给：它们包含当前这条陈述的标签。
    5f deliberately omits the party (a direct source shortcut that adds no
    context beyond the speaker's name) and the credit-history counts (they
    include the current label).

除 5f 外，只使用陈述文本，和 07 中的模型保持一致。
Apart from 5f, only the statement text is given, as for the models in notebook 07.

命令行 / Command line:
    python -m src.llm_factcheck --sample 20          # 小样本即时测试，检查输出格式 / quick sync check
    python -m src.llm_factcheck --submit no_evidence # 提交批量任务 / submit the batch
    python -m src.llm_factcheck --collect no_evidence  # 等待并收集结果 / wait and collect
    python -m src.llm_factcheck --submit no_evidence --split val   # 验证集（用于第 5c 步的组合）/ validation set
    python -m src.llm_factcheck --sample 20 --model sonnet          # 5d：更大的模型 / 5d: a larger model
    python -m src.llm_factcheck --submit no_evidence --split val --model sonnet
    python -m src.llm_factcheck --submit context --split val        # 5f
"""

import argparse
import json
import time

import numpy as np

from src import baselines as B
from src.data import PROJECT_ROOT
from src.external import load_liar

# 5a–5c 用最便宜的 Haiku 4.5；5d 换成更大的 Sonnet 5.5，检验"知识更多的模型"能否让组合继续提高
# 5a-5c use the cheapest Haiku 4.5; 5d swaps in the larger Sonnet 5.5
MODELS = {'haiku': 'claude-haiku-4-5', 'sonnet': 'claude-sonnet-5-5'}
MODEL = MODELS['haiku']
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

# 5f 的提示词：和 5a 只差一处，即说明会同时给出"谁、在哪里说的"。其余（任务、输出格式）完全相同，
# 所以 5f 和 5a 的差别只来自发言背景。
# 5f prompt: identical to 5a except that it announces who said it and where,
# so any difference from 5a comes from the context alone.
SYSTEM_CONTEXT = (
    'You are a professional political fact-checker. You will be given a statement made by a '
    'US political figure, together with who made it and where. Estimate the probability that a '
    'professional fact-checker would rate the statement as true, mostly true, or half true '
    '(rather than barely true, false, or "pants on fire"). Use your general knowledge. '
    'Reply with only an integer from 0 to 100.')

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

# Sonnet 5.5 的请求和 Haiku 有两处必须不同（否则返回 400 错误）：
#   1. 思考（thinking）默认开启。思考会消耗 max_tokens、增加费用，而这里只需要一个数字，
#      所以关掉它。Sonnet 5.5 不接受 "disabled"，最低的设置是 "between_tools"（不做扩展思考）。
#   2. 不接受非默认的 temperature，所以不传。这意味着 Sonnet 的回答有少量随机性，
#      而 Haiku 是 temperature 0。这一点作为局限记录下来。
# Sonnet 5.5 needs two differences (otherwise a 400): thinking is on by default
# and "disabled" is rejected, so the lowest setting "between_tools" is sent; a
# non-default temperature is rejected, so none is sent (its answers are therefore
# slightly random, unlike Haiku at temperature 0 - recorded as a limitation).
# 新的分词器对同样的文字产生更多 token，所以 max_tokens 留出余量
# Its tokenizer yields more tokens for the same text, so max_tokens has headroom
SONNET_THINKING = {'type': 'between_tools'}
SONNET_MAX_TOKENS = 40


def _client():
    import anthropic
    return anthropic.Anthropic()


def format_evidence(passages: list) -> str:
    """把证据段落排成带编号和文章标题的文本 / Number the passages with their article titles."""
    if not passages:
        return 'Wikipedia passages: none were found.'
    return 'Wikipedia passages:\n' + '\n'.join(
        f'[{i}] ({p["title"]}) {p["text"]}' for i, p in enumerate(passages, 1))


def format_context(row) -> str:
    """把 LIAR 的元数据排成发言背景，例如：
        Speaker: Rick Scott (Governor, Florida)
        Setting: a campaign ad
    说话人在 LIAR 中是 "rick-scott" 这样的形式，转成 "Rick Scott"。空字段直接省略，不写 "unknown"，
    以免模型把"未知"当成信息。
    Format LIAR metadata as the context of a statement; speaker slugs become
    names; empty fields are omitted rather than written as "unknown".
    """
    speaker = ' '.join(w.capitalize() for w in str(row['speaker']).split('-')) or 'unknown'
    details = ', '.join(x.strip() for x in (row['job_title'], row['state']) if str(x).strip())
    lines = [f'Speaker: {speaker}' + (f' ({details})' if details else '')]
    if str(row['context']).strip():
        lines.append(f"Setting: {str(row['context']).strip()}")
    return '\n'.join(lines)


def contexts_for(experiment: str, doc_ids, seed: int = 42) -> dict:
    """每条陈述要给模型看的发言背景：5f 是它自己的；对照组是另一条陈述的（错位分配，没有一条拿到自己的）。
    Context shown per statement: its own (5f) or another statement's (control,
    a derangement so that no statement keeps its own)."""
    from src.external import load_liar_metadata
    doc_ids = list(doc_ids)
    meta = load_liar_metadata().set_index('doc_id').loc[doc_ids]
    own = {d: format_context(meta.loc[d]) for d in doc_ids}
    if experiment == 'context':
        return own
    if experiment == 'shuffled_context':
        order = list(np.random.default_rng(seed).permutation(doc_ids))
        donor = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        return {d: own[donor[d]] for d in doc_ids}
    raise ValueError(f'unknown experiment: {experiment}')


def _params(statement: str, passages: list = None, model: str = 'haiku', context: str = None) -> dict:
    """一条陈述对应的请求参数：passages 和 context 都为 None 时是 5a；给 passages 是 5b；给 context 是 5f。
    Request parameters: neither given = 5a; passages = 5b; context = 5f.

    model='sonnet' 时，提示词和输出格式完全相同，只改模型和上面说明的两处设置。
    With model='sonnet' the prompt and output format are identical; only the
    model and the two settings explained above change.
    """
    if passages is not None and context is not None:
        raise ValueError('give either passages (5b) or context (5f), not both')
    if context is not None:
        # 背景放在陈述前面，格式和 5a 的 "Statement: ..." 保持一致 / context first, then the same "Statement: ..."
        params = {'model': MODELS[model], 'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE,
                  'system': SYSTEM_CONTEXT, 'output_config': OUTPUT_FORMAT,
                  'messages': [{'role': 'user', 'content': f'{context}\nStatement: {statement}'}]}
    elif passages is None:
        params = {'model': MODELS[model], 'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE,
                  'system': SYSTEM_NO_EVIDENCE, 'output_config': OUTPUT_FORMAT,
                  'messages': [{'role': 'user', 'content': f'Statement: {statement}'}]}
    else:
        params = {'model': MODELS[model], 'max_tokens': MAX_TOKENS_EVIDENCE, 'temperature': TEMPERATURE,
                  'system': SYSTEM_EVIDENCE, 'output_config': OUTPUT_FORMAT_EVIDENCE,
                  'messages': [{'role': 'user',
                                'content': f'Statement: {statement}\n\n{format_evidence(passages)}'}]}
    if model == 'sonnet':
        del params['temperature']
        params['thinking'] = SONNET_THINKING
        params['max_tokens'] = max(params['max_tokens'], SONNET_MAX_TOKENS)
    return params


def parse_relevant(text: str):
    """读出 evidence_relevant（True / False），没有这个字段返回 None。
    Read evidence_relevant (True / False); None if absent."""
    try:
        value = json.loads(text).get('evidence_relevant')
    except (TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def passages_for(experiment: str, doc_ids, split: str = 'test', seed: int = 42) -> dict:
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
    evidence = load_evidence(split)
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


def run_name(experiment: str, split: str = 'test', model: str = 'haiku') -> str:
    """结果文件名：测试集沿用原来的名字（llm_no_evidence 等），其他划分加上后缀（no_evidence_val）；
    Haiku 以外的模型再加上模型名（no_evidence_sonnet_val）。
    File-name stem: the test set keeps the original names, other splits get a
    suffix, and models other than Haiku add their name.

    以前所有结果都存成同一个名字；如果直接在验证集上运行，会覆盖已经保存的测试集结果。
    Previously every run used one name, so a validation run would have
    overwritten the saved test results.
    """
    if split not in ('val', 'test'):
        raise ValueError(f'unknown split: {split}')
    if model not in MODELS:
        raise ValueError(f'unknown model: {model}')
    stem = experiment if model == 'haiku' else f'{experiment}_{model}'
    return stem if split == 'test' else f'{stem}_{split}'


def _liar_sets():
    liar = load_liar()
    return {name: liar[liar['split'] == split].reset_index(drop=True)
            for name, split in (('LIAR val', 'val'), ('LIAR test', 'test'))}


def run_sample(n: int = 20, model: str = 'haiku'):
    """即时调用（非批量）测试 n 条验证集陈述：检查回答格式能否解析，并估算 token 用量。
    Synchronous check on n validation statements: parsing and token usage."""
    client = _client()
    val = _liar_sets()['LIAR val'].head(n)
    usage_in = usage_out = 0
    for _, row in val.iterrows():
        msg = client.messages.create(**_params(row['text'], model=model))
        reply = next((b.text for b in msg.content if b.type == 'text'), '')
        usage_in += msg.usage.input_tokens
        usage_out += msg.usage.output_tokens
        print(f"{row['label6']:>12} | P(true)={parse_probability(reply)} | reply={reply!r} | {row['text'][:70]}")
    print(f'\n{n} requests: {usage_in} input + {usage_out} output tokens '
          f'({usage_in / n:.0f} / {usage_out / n:.1f} per request)')
    return usage_in / n, usage_out / n


def _all_params(experiment: str, split: str, model: str) -> dict:
    """一个划分里每条陈述的请求参数 {doc_id: params}；submit 和 retry_failed 共用，保证重试的请求完全相同。
    Request parameters per statement; shared by submit and retry_failed so retries are identical."""
    df = _liar_sets()[f'LIAR {split}']
    if experiment in ('context', 'shuffled_context'):
        ctx = contexts_for(experiment, df['doc_id'])
        return {d: _params(t, None, model, context=ctx[d]) for d, t in zip(df['doc_id'], df['text'])}
    shown = passages_for(experiment, df['doc_id'], split)
    return {d: _params(t, shown[d], model) for d, t in zip(df['doc_id'], df['text'])}


def _batch_ids(meta: dict) -> list:
    """一次运行的所有批次：最初的批次，加上之后为失败的请求重新提交的批次（按时间顺序）。
    All batches of a run: the original one plus any retries of failed requests, in order."""
    return [meta['batch_id']] + meta.get('retry_batch_ids', [])


def _succeeded_ids(client, meta: dict) -> set:
    done = set()
    for bid in _batch_ids(meta):
        done |= {r.custom_id for r in client.messages.batches.results(bid) if r.result.type == 'succeeded'}
    return done


def retry_failed(experiment: str = 'no_evidence', split: str = 'test', model: str = 'haiku'):
    """只重新提交失败的请求（例如账户余额不足时被拒绝的），参数和原来完全相同；collect 会合并所有批次。
    Resubmit only the failed requests (e.g. rejected while the credit balance was
    empty) with identical parameters; collect merges all batches.

    为什么不能把失败的请求当作 0.5：在 2026-10-17 的 Sonnet test 运行中，余额耗尽导致 86 条请求失败；
    如果按"不知道"(0.5) 计入，会人为地拉低 Sonnet 的分数，影响结论。
    Why not count failures as 0.5: 86 Sonnet test requests failed when the
    balance ran out; scoring them as "don't know" would bias the result.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    name = run_name(experiment, split, model)
    path = RESULTS_DIR / f'{name}_batch.json'
    meta = json.loads(path.read_text())
    client = _client()
    for bid in _batch_ids(meta):
        if client.messages.batches.retrieve(bid).processing_status != 'ended':
            raise RuntimeError(f'batch {bid} is still running; retry after it ends')
    done = _succeeded_ids(client, meta)          # 只下载一次结果 / download the results once, not per statement
    todo = {d: p for d, p in _all_params(experiment, split, model).items() if d not in done}
    if not todo:
        print('nothing to retry')
        return None
    batch = client.messages.batches.create(requests=[
        Request(custom_id=d, params=MessageCreateParamsNonStreaming(**p)) for d, p in todo.items()])
    meta['retry_batch_ids'] = meta.get('retry_batch_ids', []) + [batch.id]
    path.write_text(json.dumps(meta, indent=2))
    print(f'resubmitted {len(todo)} failed requests as batch {batch.id}')
    return batch.id


def submit(experiment: str = 'no_evidence', split: str = 'test', model: str = 'haiku') -> str:
    """把 LIAR 一个划分的全部陈述作为一个批量任务提交，保存批次 ID。
    Submit every statement of one LIAR split as one batch; save the batch id.

    5a、5b 只在 test 上运行（它们没有需要调的东西）。val 用于之后的优化：
    组合权重、阈值等都只能在 val 上选择，所以需要 LLM 在 val 上的预测。
    5a / 5b ran on test only (nothing to tune). val is needed for later
    improvements: combination weights and thresholds may only be chosen on val.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    name = run_name(experiment, split, model)
    chosen = {k: v for k, v in _liar_sets().items() if k == f'LIAR {split}'}
    requests = [Request(custom_id=doc_id, params=MessageCreateParamsNonStreaming(**params))
                for doc_id, params in _all_params(experiment, split, model).items()]
    batch = _client().messages.batches.create(requests=requests)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 记录实际发送的设置（从第一条请求里取），而不是常量，这样 Sonnet 的不同设置也会被如实记录
    # Record the settings actually sent (taken from the first request), so Sonnet's differences are logged
    sent = requests[0]['params']
    (RESULTS_DIR / f'{name}_batch.json').write_text(json.dumps(
        {'batch_id': batch.id, 'model': sent['model'], 'n_requests': len(requests), 'sets': list(chosen),
         'system_prompt': sent['system'], 'max_tokens': sent['max_tokens'],
         'temperature': sent.get('temperature', 'not sent (model default)'),
         'thinking': sent.get('thinking', 'model default')}, indent=2))
    print(f'submitted batch {batch.id} with {len(requests)} requests')
    return batch.id


def collect(experiment: str = 'no_evidence', split: str = 'test', poll_seconds: int = 60,
            model: str = 'haiku'):
    """等待批量任务完成，取回结果，计算指标并保存（格式与 results/liar 相同）。
    Wait for the batch, fetch results, compute metrics, save (same format as results/liar)."""
    name = run_name(experiment, split, model)
    meta = json.loads((RESULTS_DIR / f'{name}_batch.json').read_text())
    client = _client()
    for bid in _batch_ids(meta):
        while True:
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == 'ended':
                break
            print(f'status {batch.processing_status}: {batch.request_counts.processing} processing', flush=True)
            time.sleep(poll_seconds)

    # 结果的返回顺序是任意的，必须按 custom_id（也就是 doc_id）对应，不能按位置。
    # 多个批次（原始 + 重试）依次合并：任何一个批次成功的回答都会被采用，只有从未成功的才记为失败。
    # Results arrive in any order: key them by custom_id (= doc_id), never by
    # position. Batches (original + retries) are merged; a request counts as
    # failed only if it never succeeded.
    replies, usage_in, usage_out, failed = {}, 0, 0, {}
    for bid in _batch_ids(meta):
        for result in client.messages.batches.results(bid):
            if result.result.type == 'succeeded':
                msg = result.result.message
                replies[result.custom_id] = next((b.text for b in msg.content if b.type == 'text'), '')
                usage_in += msg.usage.input_tokens
                usage_out += msg.usage.output_tokens
                failed.pop(result.custom_id, None)
            elif result.custom_id not in replies:
                failed[result.custom_id] = result.result.type

    eval_sets = {k: v for k, v in _liar_sets().items() if k in meta['sets']}
    probs, metrics, unparsed, relevant_share = {}, {}, {}, {}
    # 循环变量叫 set_name，不能叫 name：name 是上面的结果文件名，被覆盖后文件会存成 "llm_LIAR val"
    # The loop variable must not be called `name`: that would overwrite the
    # result-file name above (it once saved the file as "llm_LIAR val")
    for set_name, df in eval_sets.items():
        relevant = [parse_relevant(replies.get(d)) for d in df['doc_id']]
        flags = [r for r in relevant if r is not None]
        relevant_share[set_name] = round(100 * sum(flags) / len(flags), 1) if flags else None
        p = np.array([parse_probability(replies.get(d)) for d in df['doc_id']])
        unparsed[set_name] = int(np.isnan(p).sum())
        # 无法解析或失败的请求按 0.5（"不知道"）处理，并单独报告数量
        # Unparsed / failed answers count as 0.5 ("don't know") and are reported
        probs[set_name] = np.where(np.isnan(p), 0.5, p)
        metrics[set_name] = B.evaluate(df['label'], probs[set_name])

    record = {'name': f'llm_{name}', **meta, 'metrics': metrics,
              'unparsed_or_failed': unparsed, 'failed_types': failed,
              'evidence_relevant_%': relevant_share,
              'usage': {'input_tokens': usage_in, 'output_tokens': usage_out}}
    path = RESULTS_DIR / f'llm_{name}'
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
    parser.add_argument('--retry', help='experiment name whose failed requests to resubmit')
    parser.add_argument('--split', default='test', choices=['val', 'test'],
                        help='LIAR split to run on (default: test)')
    parser.add_argument('--model', default='haiku', choices=list(MODELS),
                        help='haiku (5a-5c) or sonnet (5d)')
    args = parser.parse_args()
    if args.sample:
        run_sample(args.sample, args.model)
    if args.submit:
        submit(args.submit, args.split, args.model)
    if args.retry:
        retry_failed(args.retry, args.split, args.model)
    if args.collect:
        collect(args.collect, args.split, model=args.model)


if __name__ == '__main__':
    main()
