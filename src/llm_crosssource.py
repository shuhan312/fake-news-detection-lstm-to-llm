"""
第 6 步：大模型能不能解决最初的跨来源问题？/ Step 6: can an LLM fix the original cross-source problem?

背景 / Background
    第 2–4 步中，所有在 ISOT 上训练的模型（TF-IDF、LSTM、BiLSTM、DistilBERT）在 ISOT 内部
    达到 93–99%，但在 WELFake（真新闻来自路透社以外的网站）上只有 52–59%：它们学到的是
    "路透社的写作风格"，而不是真假。第 5 步的大模型只在 LIAR（一句话的陈述）上测试过。
    Every ISOT-trained model scored 93-99% in-domain but 52-59% on WELFake,
    because it learned Reuters' style. The LLMs of step 5 were only tested on LIAR.

做法 / Method
    让 Haiku 和 Sonnet *不经过任何训练* 直接判断 WELFake test 的文章。它们没有见过 ISOT，
    所以没有机会学到"认路透社"这个捷径。和 ISOT 模型在 *同一批* 文章上比较。
    Haiku and Sonnet judge WELFake test articles zero-shot. Having never seen
    ISOT, they cannot have learned the Reuters shortcut. Compared with the ISOT
    models on the same articles.

    - 样本：WELFake test 的 1,500 篇，按"假新闻 / 纽约时报 / Breitbart / 其他真新闻"分层抽样（种子 42），
      保持各组比例。费用约为全量的 1/3。
      Sample: 1,500 test articles, stratified by publisher group (seed 42).
    - 输入：标题 + 正文前 300 个词。正文中位数 620 个词；第 2 步也发现文章开头最有信息量。
      Input: title + first 300 words (median length 620; step 2 found the start most informative).
    - 提示词保持中性：只说明"真新闻 vs 假新闻（捏造或虚假的报道）"，*不* 提示文风，以免引导模型。
      Neutral prompt: real vs fake (fabricated or false); no hint about writing style.
    - 没有需要调的东西，所以只在 test 上运行。
      Nothing is tuned, so only test is used.

需要注意 / Caveat
    WELFake 的标签也是按网站打的：某些网站的文章全算真、某些全算假。所以大模型的分数也可能部分
    来自"文风像不像假新闻网站"，而不只是核实事实。
    WELFake labels are per site too, so an LLM may partly judge site style rather than facts.

命令行 / Command line:
    python -m src.llm_crosssource --submit haiku     # 或 sonnet / or sonnet
    python -m src.llm_crosssource --collect haiku
    python -m src.llm_crosssource --retry haiku      # 只重新提交失败的请求 / resubmit failures only
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import baselines as B
from src.data import PROJECT_ROOT
from src.llm_factcheck import MAX_TOKENS, MODELS, OUTPUT_FORMAT, SONNET_MAX_TOKENS, SONNET_THINKING, \
    TEMPERATURE, parse_probability

RESULTS_DIR = PROJECT_ROOT / 'results' / 'llm'
SAMPLE_SIZE = 1500
SAMPLE_SEED = 42
MAX_WORDS = 300

SYSTEM_ARTICLE = (
    'You are a professional fact-checker. You will be given the title and the beginning of a news '
    'article. Estimate the probability that it is a real news article rather than fake news '
    '(fabricated or false stories). Use your general knowledge. Reply with only an integer from 0 to 100.')


def sample_test(n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """WELFake test 中按出版方分组分层抽样 n 篇（每组按比例抽取），结果固定、可复现。
    A stratified sample of n WELFake test articles by publisher group; fixed and reproducible."""
    from src.analysis import publisher_groups
    from src.external import load_welfake_external
    w = load_welfake_external()
    test = w[w['split'] == 'test'].reset_index(drop=True)
    groups = publisher_groups(test)
    test['group'] = groups.loc[test['doc_id']].to_numpy()
    frac = n / len(test)
    parts = [g.sample(n=int(round(len(g) * frac)), random_state=seed) for _, g in test.groupby('group')]
    return pd.concat(parts).sort_values('doc_id').reset_index(drop=True)


def article_input(title: str, text: str, max_words: int = MAX_WORDS) -> str:
    """标题 + 正文前 max_words 个词；截断时加上 "..."，让模型知道后面还有内容。
    Title plus the first max_words words of the body; "..." marks a cut."""
    words = str(text).split()
    body = ' '.join(words[:max_words]) + (' ...' if len(words) > max_words else '')
    return f'Title: {title}\n\nArticle: {body}'


def _params(title: str, text: str, model: str) -> dict:
    """和 5a / 5d 相同的设置（结构化输出、只返回一个整数）；Sonnet 同样关闭思考、不传 temperature。
    Same settings as 5a / 5d; Sonnet again with thinking off and no temperature."""
    params = {'model': MODELS[model], 'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE,
              'system': SYSTEM_ARTICLE, 'output_config': OUTPUT_FORMAT,
              'messages': [{'role': 'user', 'content': article_input(title, text)}]}
    if model == 'sonnet':
        del params['temperature']
        params['thinking'] = SONNET_THINKING
        params['max_tokens'] = SONNET_MAX_TOKENS
    return params


def _meta_path(model):
    return RESULTS_DIR / f'welfake_{model}_batch.json'


def _requests(df, model, only=None):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    return [Request(custom_id=d, params=MessageCreateParamsNonStreaming(**_params(t, x, model)))
            for d, t, x in zip(df['doc_id'], df['title'], df['text']) if only is None or d in only]


def submit(model: str) -> str:
    import anthropic
    df = sample_test()
    reqs = _requests(df, model)
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    sent = reqs[0]['params']
    _meta_path(model).write_text(json.dumps(
        {'batch_id': batch.id, 'model': sent['model'], 'n_requests': len(reqs), 'system_prompt': SYSTEM_ARTICLE,
         'max_words': MAX_WORDS, 'sample_size': SAMPLE_SIZE, 'sample_seed': SAMPLE_SEED,
         'max_tokens': sent['max_tokens'], 'temperature': sent.get('temperature', 'not sent (model default)'),
         'thinking': sent.get('thinking', 'model default')}, indent=2))
    print(f'submitted batch {batch.id} with {len(reqs)} requests')
    return batch.id


def _batch_ids(meta):
    return [meta['batch_id']] + meta.get('retry_batch_ids', [])


def retry_failed(model: str):
    """只重新提交失败的请求，参数完全相同（做法同 src/llm_factcheck.retry_failed）。
    Resubmit failed requests only, with identical parameters."""
    import anthropic
    client = anthropic.Anthropic()
    meta = json.loads(_meta_path(model).read_text())
    done = set()
    for bid in _batch_ids(meta):
        if client.messages.batches.retrieve(bid).processing_status != 'ended':
            raise RuntimeError(f'batch {bid} is still running')
        done |= {r.custom_id for r in client.messages.batches.results(bid) if r.result.type == 'succeeded'}
    df = sample_test()
    todo = set(df['doc_id']) - done
    if not todo:
        print('nothing to retry')
        return None
    batch = client.messages.batches.create(requests=_requests(df, model, only=todo))
    meta['retry_batch_ids'] = meta.get('retry_batch_ids', []) + [batch.id]
    _meta_path(model).write_text(json.dumps(meta, indent=2))
    print(f'resubmitted {len(todo)} failed requests as batch {batch.id}')
    return batch.id


def collect(model: str, poll_seconds: int = 60) -> dict:
    """等所有批次结束，合并结果；有从未成功的请求就报错（不能当作 0.5，否则会拉低分数）。
    Wait for all batches and merge; raise if any request never succeeded (never fill with 0.5)."""
    import anthropic
    client = anthropic.Anthropic()
    meta = json.loads(_meta_path(model).read_text())
    for bid in _batch_ids(meta):
        while client.messages.batches.retrieve(bid).processing_status != 'ended':
            time.sleep(poll_seconds)
    replies, usage = {}, {'input_tokens': 0, 'output_tokens': 0}
    for bid in _batch_ids(meta):
        for r in client.messages.batches.results(bid):
            if r.result.type == 'succeeded':
                msg = r.result.message
                replies[r.custom_id] = next((b.text for b in msg.content if b.type == 'text'), '')
                usage['input_tokens'] += msg.usage.input_tokens
                usage['output_tokens'] += msg.usage.output_tokens
    df = sample_test()
    missing = [d for d in df['doc_id'] if d not in replies]
    if missing:
        raise RuntimeError(f'{len(missing)} requests never succeeded; run --retry {model} first')
    prob = np.array([parse_probability(replies[d]) for d in df['doc_id']])
    unparsed = int(np.isnan(prob).sum())
    prob = np.where(np.isnan(prob), 0.5, prob)
    record = {**meta, 'usage': usage, 'unparsed': unparsed,
              'metrics': B.evaluate(df['label'], prob),
              'predicted real % by group': (pd.Series(prob > 0.5).groupby(df['group'].to_numpy()).mean()
                                            .mul(100).round(1).to_dict())}
    path = RESULTS_DIR / f'llm_welfake_{model}'
    path.with_suffix('.json').write_text(json.dumps(record, indent=2))
    np.savez_compressed(path.with_suffix('.npz'), prob=prob, doc_id=df['doc_id'].to_numpy(dtype=str),
                        reply=np.array([replies[d] for d in df['doc_id']], dtype=str))
    print(f"{model}: AUC {record['metrics']['AUC']} accuracy {record['metrics']['accuracy %']} | "
          f"unparsed {unparsed} | by group {record['predicted real % by group']} | tokens {usage}")
    return record


# 记忆检查说明 / Note on a memorisation check
# 尝试过文本污染检测的常用方法（给出标题和前 60 个词，让模型逐字续写，看是否和原文有连续 8 个词相同）。
# 在 4 篇纽约时报文章上的小样本测试中，Sonnet 每次都拒绝复现已发表的原文，改写成自己的续写，
# 所以这个测试无论模型是否"见过"文章都会显示"没记住"，结果没有意义，因此没有全量运行。
# The standard continuation test (reproduce the next words verbatim) was tried
# on 4 NYT articles: Sonnet declined every time to reproduce published text and
# wrote its own continuation, so the test cannot detect memorisation and was
# not run in full.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 6: zero-shot LLMs on the WELFake cross-source test.')
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
