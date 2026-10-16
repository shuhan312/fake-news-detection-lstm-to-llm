"""
第 5 步 5b：为 LIAR 陈述检索维基百科证据 / Step 5b: Wikipedia evidence for LIAR statements.

为什么需要证据 / Why evidence
    5a 表明：不给证据时，Claude Haiku 只有 AUC 0.655，和只看文字的模型差不多。一句话是真是假，
    答案通常不在这句话里，而在外部事实里。5b 先检索维基百科，再把证据和陈述一起交给模型。
    Without evidence Haiku reaches AUC 0.655, like text-only models: truth usually
    depends on outside facts. 5b retrieves Wikipedia passages and gives them to the model.

为什么用维基百科 / Why Wikipedia
    免费；而且比网络搜索更不容易泄漏答案 —— 在网上搜一条 LIAR 陈述，第一条结果很可能就是
    PolitiFact 对它的核查文章。
    Free, and less leaky than web search, where the first hit for a LIAR statement
    is often PolitiFact's own fact-check of it.

流程 / Pipeline (per statement)
    1. 让 Claude Haiku 为每条陈述写 2 个简短的维基百科搜索词（只看陈述本身，并被要求不判断真假）；
       拿不到时退回规则方法（专有名词 + 内容词）
       Claude Haiku writes 2 short Wikipedia search queries per statement (it sees only
       the statement and is told not to judge it); rule-based queries are the fallback
       （实测：直接用整句搜索效果很差；规则方法也会出错，例如把句首的 "Having" 当成专有名词，
        搜到 SQL 的 HAVING 语句，或把 "Americans working" 搜成一支叫 Hard Working Americans 的乐队）
    2. 两路检索结果合并去重，最多取 MAX_PAGES 篇文章 / merge, keep up to MAX_PAGES articles
    3. 把文章拆成段落，用 TF-IDF 选出和陈述最相关的 N_PASSAGES 段
       split into paragraphs; TF-IDF picks the N_PASSAGES most similar to the statement
    4. 过滤可能泄漏答案的段落（见 is_leaky）/ drop paragraphs that could leak the verdict

检索结果缓存在 data/external/wiki_evidence_cache.json（不进 git），最终用到的证据另存到
results/llm/evidence_liar_<split>.json.gz，保证结果可以复现（维基百科内容采用 CC BY-SA 许可）。
Retrieval is cached in data/external/ (not in git); the evidence actually used is
saved to results/llm/ so results are reproducible (Wikipedia content: CC BY-SA).
"""

import gzip
import json
import re
import time
import urllib.parse
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import PROJECT_ROOT
from src.external import EXTERNAL_DIR
from src.step4 import FUNCTION_WORDS

API_URL = 'https://en.wikipedia.org/w/api.php'
# 维基百科要求请求带上能识别来源的 User-Agent；这里用项目地址，不包含任何个人信息
# Wikipedia asks for an identifying User-Agent; the project URL is used, no personal data
USER_AGENT = ('fake-news-detection-lstm-to-llm research '
              '(https://github.com/shuhan312/fake-news-detection-lstm-to-llm)')
REQUEST_PAUSE = 0.2          # 每次请求之间停 0.2 秒，避免给维基百科服务器添负担 / be polite
WORKERS = 4                  # 同时进行的检索数 / parallel retrievals
CACHE_PATH = EXTERNAL_DIR / 'wiki_evidence_cache.json'
EVIDENCE_DIR = PROJECT_ROOT / 'results' / 'llm'      # evidence_liar_<split>.json.gz

# 这些数值在看到任何测试结果 *之前* 就定好了，之后不会根据测试集调整
# Fixed BEFORE seeing any test result; never tuned on the test set
MAX_PAGES = 3                # 每条陈述最多用几篇文章 / articles per statement
N_PASSAGES = 3               # 最终给模型几段证据 / passages given to the model
MAX_PASSAGE_WORDS = 150      # 每段最多保留多少词 / words kept per passage
MIN_PARAGRAPH_WORDS = 20     # 太短的段落（标题、列表项）不用 / skip headings and list items
CONTENT_QUERY_WORDS = 6      # 内容词检索最多用几个词 / words in the content-word query

# 检索词里要去掉的、没有信息量的词（除了虚词表之外）
# Uninformative words removed from queries (besides the function-word list)
QUERY_NOISE = frozenset('says said say percent people time year years today day days number '
                        'gov sen rep mr mrs ms dr'.split())

# 泄漏规则 1：提到事实核查机构或它们的评级用语
# Leak rule 1: mentions of fact-checkers or their rating vocabulary
_RE_FACTCHECK = re.compile(
    r'(?i)politifact|fact[- ]?check|truth-o-meter|pants on fire|snopes|factcheck\.org|'
    r'pinocchio|mostly false|half true|mostly true')
# 泄漏规则 2：段落几乎原样复述了这条陈述，并且下了"真 / 假"的判断
# Leak rule 2: the paragraph restates the claim AND passes a verdict on it
_RE_VERDICT = re.compile(
    r'(?i)\bfalse(?:ly)?\b|\buntrue\b|\bdebunk|\bmisleading\b|\binaccurate\b|\bbaseless\b|'
    r'\bunfounded\b|\bno evidence\b|\bconspiracy theor')
RESTATEMENT_OVERLAP = 0.6    # 陈述的内容词有 60% 以上出现在段落中，视为"复述" / counts as restating

# 参考文献条目（书目、链接存档）不是证据。小样本中出现过
# "Rescher, Nicholas (1998). Predicting the Future ... ISBN 978-..." 这样的段落
# Reference-list entries are not evidence (a sample picked "... ISBN 978-...")
_RE_REFERENCE = re.compile(r'\bISBN\b|\bdoi:|\bRetrieved\b|\bArchived from\b|\(Webpage removed\)')
# 作者-日期格式的参考文献，例如 'Kessler, Glenn (January 21, 2013). "The stale claim that ..."'。
# 这条正是从测试集证据审计中发现的：它是《华盛顿邮报》核查文章的标题，直接泄漏了结论。
# Author-date citations such as 'Kessler, Glenn (January 21, 2013). "The stale claim ..."' -
# found by the test-evidence audit: a fact-check headline that leaks the verdict.
_RE_CITATION = re.compile(
    r"^[A-Z][A-Za-z'\-]+, [A-Z][A-Za-z.'\- ]{1,40} \((?:[A-Z][a-z]+ )?(?:\d{1,2}, )?\d{4}\)")

# 专门收录"错误说法、阴谋论、谣言"的文章，整篇的定位就是"这些是假的"，例如
# "False or misleading statements by Donald Trump"、"Barack Obama citizenship conspiracy theories"。
# 证据来自这类文章，等于用标题告诉模型答案，所以整篇不用。
# Articles that catalogue false claims, conspiracy theories or rumours (e.g. "False or
# misleading statements by Donald Trump"): their very title tells the model the verdict.
_RE_VERDICT_ARTICLE = re.compile(
    r'(?i)\bfalse\b|misleading|misinformation|disinformation|conspiracy theor|hoax|'
    r'fact[- ]?check|debunk|urban legend|\bmyths?\b|\brumou?rs?\b|propaganda|fake news')


# ─────────────────────────────────────────────────────────────────────────────
# 检索词 / Queries
# ─────────────────────────────────────────────────────────────────────────────

def _content_words(text: str) -> list:
    """陈述中的内容词（去掉虚词、数字和没有信息量的词），按原顺序。
    Content words of a statement (no function words, numbers or noise), in order."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", text)
    return [w for w in words if w.lower() not in FUNCTION_WORDS and w.lower() not in QUERY_NOISE]


def _entity_spans(text: str) -> list:
    """连续的首字母大写词组，例如 "Terry Branstad"、"Social Security"，去掉首尾的虚词。
    Runs of capitalised words such as "Terry Branstad", trimmed of filler words.

    句号会切断词组，所以 "Iowa Gov. Terry Branstad" 得到 "Iowa" 和 "Terry Branstad"。
    A full stop ends a span, so "Iowa Gov. Terry Branstad" gives "Iowa" and "Terry Branstad".
    """
    spans = []
    for span in re.findall(r"[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*", text):
        words = span.split()
        while words and (words[0].lower() in FUNCTION_WORDS or words[0].lower() in QUERY_NOISE):
            words.pop(0)
        while words and (words[-1].lower() in FUNCTION_WORDS or words[-1].lower() in QUERY_NOISE):
            words.pop()
        if words and len(' '.join(words)) > 2:
            spans.append(' '.join(words))
    return spans


def build_queries(statement: str) -> dict:
    """两个检索词：专有名词（最长的两个）和内容词（前 CONTENT_QUERY_WORDS 个）。
    Two queries: the two longest proper-noun spans, and the first content words."""
    spans = sorted(set(_entity_spans(statement)), key=len, reverse=True)[:2]
    return {'entities': ' '.join(spans),
            'content': ' '.join(_content_words(statement)[:CONTENT_QUERY_WORDS])}


# ─────────────────────────────────────────────────────────────────────────────
# 用 LLM 生成检索词 / LLM-written search queries
# ─────────────────────────────────────────────────────────────────────────────

QUERY_MODEL = 'claude-haiku-4-5'
QUERY_CACHE_PATH = EXTERNAL_DIR / 'wiki_query_cache.json'

# 只让模型写搜索词，明确要求它不要判断真假：生成搜索词这一步不能把"答案"带进证据里
# The model only writes search queries and is told not to judge the statement,
# so query writing cannot smuggle a verdict into the evidence
QUERY_SYSTEM = (
    'You help a fact-checker find background information on Wikipedia. Given a statement by a US '
    'political figure, write two short Wikipedia search queries (2-6 words each) that would find '
    'the articles containing the facts needed to check it: the people, places, programs, laws or '
    'statistics it is about. Do not judge whether the statement is true.')
QUERY_FORMAT = {'format': {'type': 'json_schema', 'schema': {
    'type': 'object',
    'properties': {'queries': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['queries'], 'additionalProperties': False}}}


def query_params(statement: str) -> dict:
    """生成检索词的请求参数（最便宜的 Haiku，temperature 0）/ Request parameters for query writing."""
    return {'model': QUERY_MODEL, 'max_tokens': 60, 'temperature': 0.0, 'system': QUERY_SYSTEM,
            'output_config': QUERY_FORMAT,
            'messages': [{'role': 'user', 'content': f'Statement: {statement}'}]}


def parse_queries(text: str) -> list:
    """从 {"queries": [...]} 中取出最多 2 个非空检索词；格式不对返回空列表。
    Up to 2 non-empty queries from {"queries": [...]}; [] if malformed."""
    try:
        queries = json.loads(text)['queries']
    except (TypeError, ValueError, KeyError):
        return []
    # 必须是列表：如果是一个字符串，逐个遍历会把它拆成单个字母
    # Must be a list: iterating a plain string would split it into letters
    if not isinstance(queries, list):
        return []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:2]


def generate_queries(df, batch: bool = True, poll_seconds: int = 30) -> dict:
    """为一批陈述生成检索词，结果缓存到 data/external/wiki_query_cache.json。
    Generate queries for many statements (cached).

    batch=True 时用 Batch API（五折，适合全量）；False 时逐条即时调用（适合小样本）。
    batch=True uses the Batch API (half price, for full runs); False calls synchronously.
    """
    import anthropic
    client = anthropic.Anthropic()
    cache = json.loads(QUERY_CACHE_PATH.read_text()) if QUERY_CACHE_PATH.exists() else {}
    todo = {d: t for d, t in zip(df['doc_id'], df['text']) if d not in cache}
    usage = {'input_tokens': 0, 'output_tokens': 0}

    if todo and not batch:
        for doc_id, text in todo.items():
            msg = client.messages.create(**query_params(text))
            cache[doc_id] = parse_queries(next((b.text for b in msg.content if b.type == 'text'), ''))
            usage['input_tokens'] += msg.usage.input_tokens
            usage['output_tokens'] += msg.usage.output_tokens
    elif todo:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        job = client.messages.batches.create(requests=[
            Request(custom_id=d, params=MessageCreateParamsNonStreaming(**query_params(t)))
            for d, t in todo.items()])
        print(f'query batch {job.id}: {len(todo)} requests', flush=True)
        while client.messages.batches.retrieve(job.id).processing_status != 'ended':
            time.sleep(poll_seconds)
        # 结果顺序任意，按 custom_id 对应 / results arrive in any order: key by custom_id
        for result in client.messages.batches.results(job.id):
            if result.result.type == 'succeeded':
                msg = result.result.message
                cache[result.custom_id] = parse_queries(
                    next((b.text for b in msg.content if b.type == 'text'), ''))
                usage['input_tokens'] += msg.usage.input_tokens
                usage['output_tokens'] += msg.usage.output_tokens

    QUERY_CACHE_PATH.write_text(json.dumps(cache))
    print('query-generation tokens:', usage, flush=True)
    return {d: cache.get(d, []) for d in df['doc_id']}


# ─────────────────────────────────────────────────────────────────────────────
# 维基百科接口 / Wikipedia API
# ─────────────────────────────────────────────────────────────────────────────

def _api(params: dict, retries: int = 3) -> dict:
    """调用维基百科接口；失败时等待后重试（1、2、4 秒）。
    Call the Wikipedia API, retrying with 1 / 2 / 4 s back-off."""
    url = API_URL + '?' + urllib.parse.urlencode({**params, 'format': 'json'})
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_PAUSE)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def search_titles(query: str, limit: int) -> list:
    if not query:
        return []
    data = _api({'action': 'query', 'list': 'search', 'srsearch': query, 'srlimit': limit})
    return [hit['title'] for hit in data['query']['search']]


# 文章全文缓存：很多陈述会检索到同一篇文章（例如 "United States"、"Donald Trump"），
# 同一篇文章只下载一次。小样本中 20 条陈述用了 217 秒，大部分时间花在重复下载大文章上。
# Article cache: many statements hit the same articles ("United States", "Donald
# Trump"); each is downloaded once. A 20-statement sample took 217 s, mostly
# re-downloading large articles.
_PAGE_CACHE = {}


def page_text(title: str) -> str:
    """一篇文章的纯文本（不含维基标记），同一篇只下载一次 / Plain text of an article, fetched once."""
    if title not in _PAGE_CACHE:
        data = _api({'action': 'query', 'prop': 'extracts', 'explaintext': 1, 'titles': title,
                     'redirects': 1})
        pages = data['query']['pages']
        _PAGE_CACHE[title] = next(iter(pages.values())).get('extract', '') or ''
    return _PAGE_CACHE[title]


# ─────────────────────────────────────────────────────────────────────────────
# 选段落和过滤 / Passage selection and filtering
# ─────────────────────────────────────────────────────────────────────────────

def is_leaky(paragraph: str, statement: str) -> str:
    """判断段落是否可能泄漏答案；返回原因（'factcheck' / 'restated_verdict'），否则返回 ''。
    Return why a paragraph could leak the verdict ('factcheck' / 'restated_verdict'), or ''."""
    if _RE_FACTCHECK.search(paragraph):
        return 'factcheck'
    claim_words = {w.lower() for w in _content_words(statement)}
    if claim_words and _RE_VERDICT.search(paragraph):
        para_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", paragraph)}
        if len(claim_words & para_words) / len(claim_words) >= RESTATEMENT_OVERLAP:
            return 'restated_verdict'
    return ''


def select_passages(statement: str, pages: dict) -> tuple:
    """从文章里选出和陈述最相关、且不泄漏答案的 N_PASSAGES 段。
    Pick the N_PASSAGES paragraphs most similar to the statement that do not leak.

    Returns:
        (passages, removed)：passages 是 [{'title', 'text'}]；removed 统计每类被过滤的段落数
    """
    candidates = []
    removed = {'factcheck': 0, 'restated_verdict': 0, 'verdict_article': 0, 'reference': 0}
    for title, text in pages.items():
        for para in text.split('\n'):
            para = para.strip()
            if len(para.split()) < MIN_PARAGRAPH_WORDS:
                continue
            if _RE_VERDICT_ARTICLE.search(title):          # 整篇是"错误说法清单" / verdict catalogue
                removed['verdict_article'] += 1
                continue
            if _RE_REFERENCE.search(para) or _RE_CITATION.search(para):
                removed['reference'] += 1
                continue
            reason = is_leaky(para, statement)
            if reason:
                removed[reason] += 1
                continue
            candidates.append({'title': title, 'text': ' '.join(para.split()[:MAX_PASSAGE_WORDS])})
    if not candidates:
        return [], removed
    # 用 TF-IDF 计算每段和陈述的相似度，选最高的几段 / TF-IDF similarity to the statement
    vec = TfidfVectorizer(stop_words=list(FUNCTION_WORDS), sublinear_tf=True)
    matrix = vec.fit_transform([statement] + [c['text'] for c in candidates])
    scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    top = np.argsort(-scores)[:N_PASSAGES]
    return [dict(candidates[i], score=round(float(scores[i]), 3)) for i in top if scores[i] > 0], removed


def retrieve(statement: str, llm_queries: list = None) -> dict:
    """一条陈述的完整检索流程 / Full retrieval for one statement.

    有 LLM 生成的检索词就用它们（每个取前 2 篇文章），否则退回规则检索词。
    Uses the LLM-written queries when available (top 2 articles each), else the rules.
    """
    if llm_queries:
        queries = {'llm': llm_queries}
        hits = [t for q in llm_queries for t in search_titles(q, 2)]
    else:
        queries = build_queries(statement)
        hits = search_titles(queries['entities'], 2) + search_titles(queries['content'], 2)
    titles = []
    for title in hits:
        if title not in titles:
            titles.append(title)
    titles = titles[:MAX_PAGES]
    pages = {t: page_text(t) for t in titles}
    passages, removed = select_passages(statement, pages)
    return {'queries': queries, 'titles': titles, 'passages': passages, 'removed': removed}


def retrieve_all(df, llm_queries: dict = None, progress_every: int = 50) -> dict:
    """为一批陈述检索证据，结果写入缓存；已在缓存中的直接跳过（可以中断后继续）。
    Retrieve evidence for many statements, caching results; cached ones are skipped.

    llm_queries：generate_queries() 的结果 {doc_id: [检索词]}；None 时全部用规则检索词。
    llm_queries: output of generate_queries(); None means rule-based queries throughout.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    todo = [(d, t) for d, t in zip(df['doc_id'], df['text']) if d not in cache]

    def job(doc_id, text):
        return doc_id, retrieve(text, (llm_queries or {}).get(doc_id))

    # 同时最多 WORKERS 个请求：比逐条快很多，但并发不大，不会给维基百科服务器造成负担
    # At most WORKERS requests in flight: much faster than serial, still light on Wikipedia
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(job, d, t) for d, t in todo]
        for future in as_completed(futures):
            try:
                doc_id, result = future.result()
                cache[doc_id] = result
            except Exception as e:                   # 网络失败的条目不写入缓存，下次重试
                print(f'failed: {e}', flush=True)    # failed items are retried next time
                continue
            done += 1
            if done % progress_every == 0:
                CACHE_PATH.write_text(json.dumps(cache))
                print(f'{done}/{len(todo)} retrieved', flush=True)
    CACHE_PATH.write_text(json.dumps(cache))
    return {d: cache[d] for d in df['doc_id'] if d in cache}


def _evidence_path(split: str):
    # 按划分分别保存，避免为验证集准备证据时覆盖测试集的证据
    # One file per split, so preparing val never overwrites the test evidence
    return EVIDENCE_DIR / f'evidence_liar_{split}.json.gz'


def save_evidence(evidence: dict, split: str):
    """把实际使用的证据压缩保存到 results/llm/，用于复现 / Save the evidence used, for reproducibility."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(_evidence_path(split), 'wt', encoding='utf-8') as f:
        json.dump(evidence, f)


def load_evidence(split: str = 'test') -> dict:
    with gzip.open(_evidence_path(split), 'rt', encoding='utf-8') as f:
        return json.load(f)


def prepare(split: str = 'test'):
    """为 LIAR 的一个划分准备证据：生成检索词（Batch API）→ 检索维基百科 → 保存证据。
    Prepare evidence for one LIAR split: queries (Batch API) → Wikipedia → save.

    所有步骤都有缓存，中断后重新运行会从断点继续。
    Every step is cached, so re-running resumes where it stopped.
    """
    from src.external import load_liar

    liar = load_liar()
    df = liar[liar['split'] == split].reset_index(drop=True)
    queries = generate_queries(df, batch=True)
    evidence = retrieve_all(df, queries)
    save_evidence(evidence, split)

    n_passages = [len(v['passages']) for v in evidence.values()]
    removed = {k: sum(v['removed'].get(k, 0) for v in evidence.values())
               for k in ('factcheck', 'restated_verdict', 'verdict_article', 'reference')}
    print(f'{len(evidence)}/{len(df)} statements with evidence records; '
          f'{sum(n == 0 for n in n_passages)} without any passage; removed as leaky: {removed}', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Step 5b: Wikipedia evidence for LIAR.')
    parser.add_argument('--prepare', default='test', choices=['val', 'test'])
    prepare(parser.parse_args().prepare)
