"""
外部数据集 / External datasets for cross-dataset evaluation.

为什么需要外部数据集？/ Why external data?
    04_leakage_audit 表明：ISOT 里真新闻全部来自路透社，模型可以靠"路透社文风"拿到高分。
    要知道模型学到的是不是文风，就必须在 *来源不同* 的数据上测试。
    04_leakage_audit showed ISOT's real news is all Reuters, so a model can
    score highly on "Reuters style" alone. Only data from DIFFERENT sources
    can tell us whether that is what it learned.

这里提供两个数据集 / Two datasets
    1. WELFake 去掉 ISOT 后的部分（主要外部测试集）
       WELFake minus ISOT (primary external test set)
       - 全文新闻，真新闻来自《纽约时报》、Breitbart 等，*不是* 路透社
         full articles; real news from NYT, Breitbart etc. — NOT Reuters
       - 许可 CC-BY-4.0（需注明出处）/ licence CC-BY-4.0 (attribution)
       - 注意：标签同样是按来源网站打的，不是逐篇核查
         caveat: labels are still assigned per source site, not per article
    2. LIAR（次要外部测试集 / secondary）
       - PolitiFact 逐条核查过的短陈述，标签直接反映真假
         short statements individually fact-checked by PolitiFact
       - 只有一句话，和长文章格式差别大，所以只作补充
         one sentence each — very different format, so supplementary only
       - 许可：仅限研究用途 / licence: research use only

数据文件放在 data/external/（不提交到 git，运行时自动下载并校验 MD5）。
Files live in data/external/ (not committed; downloaded and MD5-checked on demand).
"""

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.data import (MIN_BODY_WORDS, NEAR_DUP_LEAD_WORDS, NEAR_DUP_MIN_TITLE_WORDS,
                      PROJECT_ROOT, clean_text, clean_title, load_dataset, to_model_text)

EXTERNAL_DIR = PROJECT_ROOT / 'data' / 'external'

# ─────────────────────────────────────────────────────────────────────────────
# 数据来源 / Sources
# ─────────────────────────────────────────────────────────────────────────────

# WELFake：Verma, Agrawal & Prodan (2021)，Zenodo DOI 10.5281/zenodo.4561253，CC-BY-4.0
# MD5 来自 Zenodo 官方 API / MD5 published by the Zenodo API
WELFAKE_URL = 'https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1'
WELFAKE_FILE = 'WELFake_Dataset.csv'
WELFAKE_MD5 = '73c9675a4b3d09f86a6933d0b8d7d908'

# LIAR：Wang (2017), ACL。官方没有公布校验值，这里记录的是 2026-10-14 下载时的 MD5，
# 用来保证以后下载到的是同一份文件。
# LIAR: Wang (2017), ACL. No official checksum; this is the MD5 observed when
# downloaded on 2026-10-14, so later downloads are verified to be the same file.
LIAR_URL = 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip'
LIAR_FILE = 'liar_dataset.zip'
LIAR_MD5 = '437541ca9957e993634f9c1e71a26748'

# 外部数据处理结果的缓存 / Cache of the processed external data
WELFAKE_CACHE = EXTERNAL_DIR / 'welfake_external_clean.csv'
WELFAKE_CACHE_META = EXTERNAL_DIR / 'welfake_external_clean_meta.json'


# ─────────────────────────────────────────────────────────────────────────────
# 下载与校验 / Download and verify
# ─────────────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    """分块计算文件的 MD5，避免把 245MB 一次性读进内存。
    MD5 of a file, computed in chunks so a 245 MB file never sits in memory at once."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, filename: str, md5: str = None) -> Path:
    """下载文件到 data/external/；已存在且校验通过就跳过。
    Download into data/external/; skip if the file already exists and verifies.

    urllib 会自动使用环境变量里的 HTTPS_PROXY，所以在需要代理的网络下也能用。
    urllib honours the HTTPS_PROXY environment variable, so it works behind a proxy.

    Raises:
        ValueError: 下载的文件 MD5 和预期不一致（文件损坏或被替换）。
                    the downloaded file's MD5 does not match (corrupt or replaced).
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXTERNAL_DIR / filename
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return dest

    print(f'Downloading {filename} ...')
    tmp = dest.with_suffix(dest.suffix + '.part')   # 先写临时文件，下载完整后再改名
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, 'wb') as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)

    actual = _md5(tmp)
    if md5 is not None and actual != md5:
        tmp.unlink()
        raise ValueError(f'MD5 mismatch for {filename}: expected {md5}, got {actual}')
    tmp.rename(dest)
    print(f'Saved {dest.name} (md5 {actual})')
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# WELFake
# ─────────────────────────────────────────────────────────────────────────────

# 标题末尾的出版方后缀，例如 " - The New York Times"。在 WELFake 去掉 ISOT 后的部分里，
# 63% 的真新闻标题带这种后缀，假新闻只有 2.6%，和 ISOT 的 "[VIDEO]" 一样是来源标签。
# 这里只列出数据中实际出现 5 次以上的出版方名称，避免误删 "Boiler Room #62 - Fatal Illusions"
# 这类标题本身的副标题。
# Publisher suffixes such as " - The New York Times": 63% of real vs 2.6% of fake
# titles in the non-ISOT part of WELFake — a source tag like ISOT's "[VIDEO]".
# Only publishers seen 5+ times are listed, so genuine subtitles
# ("Boiler Room #62 - Fatal Illusions") are kept.
# 判断是否为英文：常见英文虚词在全部词中的占比。英文新闻的中位数约为 36%，
# 低于 10% 的基本都是其他语言，或者清洗后只剩标点的文本（例如俄语原文的字母被去掉了）。
# English check: share of tokens that are common English function words.
# English news has a median of ~36%; below 10% is almost always another
# language, or text left as bare punctuation after cleaning (e.g. Cyrillic).
ENGLISH_FUNCTION_WORDS = frozenset(
    'the of and to a in is that for it on was with as he said by at from his be have '
    'are this not but they we you i an or which their has had were been will would '
    'there who what all can more one about'.split())
MIN_ENGLISH_SHARE = 0.10

_RE_PUBLISHER_SUFFIX = re.compile(
    r'\s+-\s+(?:The New York Times|NYTimes\.com|Breitbart|Russia News Now|RT Arabic|'
    r'USAPoliticsNow|Morning News USA|New Earth Media)\s*$')


def load_welfake_raw() -> pd.DataFrame:
    """读取原始 WELFake，并把标签改成本项目的约定（1 = 真，0 = 假）。
    Read raw WELFake and convert labels to this project's convention (1 = real, 0 = fake).

    ⚠️ 标签方向 / Label direction
        Zenodo 页面写的是 "0 = fake and 1 = real"，但实际数据 *正好相反*。
        验证方法：WELFake 包含 ISOT 的文章，其中 21,416 篇 ISOT 真新闻全部标为 0，
        22,641 篇 ISOT 假新闻标为 1（只有 1 篇例外）。论文中"37,106 篇假新闻"的数量
        也和 label == 1 的数量一致。所以这里用 1 - label 转换。
        The Zenodo page says "0 = fake and 1 = real" but the file is the OPPOSITE:
        all 21,416 ISOT-real articles inside WELFake carry 0 and 22,641 ISOT-fake
        ones carry 1 (one exception); the paper's "37,106 fake" also equals the
        count of label == 1. Hence 1 - label. verify_welfake_label_direction()
        re-checks this every time.
    """
    path = download(WELFAKE_URL, WELFAKE_FILE, WELFAKE_MD5)
    df = pd.read_csv(path, keep_default_na=False)
    df = df.rename(columns={'Unnamed: 0': 'row'})
    df['doc_id'] = [f'welfake_{i:05d}' for i in df['row']]
    df['label'] = 1 - df['label'].astype(int)
    return df[['doc_id', 'label', 'title', 'text']]


def verify_welfake_label_direction(welfake: pd.DataFrame, isot_raw: pd.DataFrame) -> pd.DataFrame:
    """用 WELFake 里包含的 ISOT 文章检查标签方向是否正确。
    Check the label direction using the ISOT articles that WELFake contains.

    对每篇能在 ISOT 里找到原文的 WELFake 文章，比较两边的标签。转换正确时，两边应该几乎完全一致。
    For every WELFake article found in ISOT, compare the two labels; after a
    correct conversion they should agree almost perfectly.

    Returns:
        交叉表：行是 ISOT 标签，列是转换后的 WELFake 标签。
        Crosstab of ISOT label (rows) vs converted WELFake label (columns).

    Raises:
        AssertionError: 一致率低于 99.9%，说明标签方向可能又反了。
                        agreement below 99.9% — the direction is probably wrong.
    """
    isot_key = _text_key(isot_raw['text'])
    isot_label = (isot_raw.assign(k=isot_key)
                          .loc[lambda d: d['k'].str.len() > 50]
                          .drop_duplicates('k').set_index('k')['label'])
    matched_isot_label = _text_key(welfake['text']).map(isot_label)
    both = welfake.assign(isot_label=matched_isot_label).dropna(subset=['isot_label'])

    agreement = (both['isot_label'] == both['label']).mean()
    assert agreement > 0.999, f'WELFake/ISOT label agreement only {agreement:.3%}'
    return pd.crosstab(both['isot_label'].astype(int).rename('ISOT label'),
                       both['label'].rename('WELFake label (converted)'))


def build_welfake_external(verbose: bool = True):
    """构建"WELFake 去掉 ISOT"的外部测试集，处理方式和 ISOT 完全一致。
    Build "WELFake minus ISOT" with exactly the same processing as ISOT.

    步骤 / Steps
        1. 读取并转换标签方向，再用 ISOT 重叠部分验证一遍
           load, convert labels, re-verify against the ISOT overlap
        2. 用 src/data.py 的同一套规则清洗标题和正文，再删除出版方后缀
           clean with the SAME rules as src/data.py, then strip publisher suffixes
        3. 和 ISOT 管线一样：删除正文少于 MIN_BODY_WORDS 的文章、规范化后完全重复的文章；
           另外删除非英文文章
           as in the ISOT pipeline: drop short bodies and exact duplicates; also
           drop non-English articles
        4. 删除所有和 ISOT 有关联的文章：先用原文比对删掉副本，再比较五种指纹
           （完全相同、标题、开头 25 词、结尾 25 词、中间 25 词）。
           比 ISOT 内部划分更严格：只要有一个指纹相同就删除，因为外部测试集里
           哪怕混入一篇 ISOT 文章，都会高估跨数据集表现。
           drop anything linked to ISOT: raw-text copies first, then five
           fingerprints (exact, title, 25-word lead, tail, middle). Stricter than
           ISOT's internal split: one shared fingerprint is enough, because any
           ISOT article in the external set inflates the result.

    Returns:
        (df, report)：列与 load_dataset() 一致（没有 group_id / split）。
        (df, report): same columns as load_dataset() minus group_id / split.
    """
    from src.data import load_raw   # 只在构建时需要 / only needed when building

    report = {}
    raw = load_welfake_raw()
    isot_raw = load_raw()
    report['raw_total'] = len(raw)
    report['label_check'] = verify_welfake_label_direction(raw, isot_raw).to_dict()

    # 先用原文比对，删掉明显是 ISOT 副本的文章（约 4.4 万篇）。这一步很便宜，
    # 可以避免对它们做耗时的清洗。
    # First drop obvious ISOT copies (~44k) by raw-text match — cheap, and it
    # saves cleaning them for nothing.
    isot_raw_keys = _key_set(_text_key(isot_raw['text']).where(lambda k: k.str.len() > 50))
    is_isot_copy = _text_key(raw['text']).isin(isot_raw_keys)
    report['removed_isot_copies_raw_match'] = _count_by_label(raw[is_isot_copy])
    df = raw[~is_isot_copy].copy()

    df['title'] = df['title'].map(clean_title).str.replace(_RE_PUBLISHER_SUFFIX, '', regex=True)
    df['text'] = df['text'].map(clean_text)
    df['content'] = (df['title'] + ' ' + df['text']).str.strip()

    short = df['text'].str.split().str.len() < MIN_BODY_WORDS
    report['removed_short_body'] = _count_by_label(df[short])
    df = df[~short].copy()

    df['model_text'] = to_model_text(df['content'])

    # 非英文文章：几乎全是假新闻（俄语、西班牙语、德语等转载）。保留的话，
    # "不是英文"本身就成了假新闻信号，而 ISOT 模型从没见过非英文，测出来的结果没有意义。
    # Non-English articles are almost all fake (Russian / Spanish / German
    # re-posts). Kept, "not English" would itself signal fake, and the ISOT model
    # has never seen non-English text, so the result would be meaningless.
    non_english = _english_share(df['model_text']) < MIN_ENGLISH_SHARE
    report['removed_non_english'] = _count_by_label(df[non_english])
    df = df[~non_english]

    conflicting = df.groupby('model_text')['label'].transform('nunique') > 1
    report['removed_conflicting_label_rows'] = int(conflicting.sum())
    df = df[~conflicting]
    dup = df.duplicated('model_text', keep='first')
    report['removed_exact_duplicates'] = _count_by_label(df[dup])
    df = df[~dup].copy()

    # 剩下的再和 *清洗后* 的 ISOT 比较四种指纹，任何一个相同就删除。
    # 能抓到原文略有差异、但清洗后相同的副本，以及换了开头或标题的近似重复。
    # Then compare four fingerprints against CLEANED ISOT; any match removes the
    # article. This catches copies whose raw text differed slightly, and
    # near-duplicates with a changed lead or title.
    # 注意：清洗后的 ISOT 是 load_dataset() 的全部 37,982 篇（train + val + test），
    # 不只是 train。外部测试集必须和 ISOT 的 *任何* 部分都没有重叠。
    # Note: compared against ALL of cleaned ISOT (train + val + test), not just
    # train — the external set must share nothing with any part of ISOT.
    isot_clean = load_dataset(verbose=False)
    isot_keys = _fingerprints(isot_clean['title'], isot_clean['text'], isot_clean['model_text'])
    ext_keys = _fingerprints(df['title'], df['text'], df['model_text'])
    linked = pd.Series(False, index=df.index)
    for name in ext_keys:
        linked |= ext_keys[name].isin(_key_set(isot_keys[name]))
    report['removed_isot_overlap_fingerprint'] = _count_by_label(df[linked])
    df = df[~linked].reset_index(drop=True)

    report['final_total'] = len(df)
    report['final_by_label'] = _count_by_label(df)
    if verbose:
        print(json.dumps(report, indent=2, default=str))
    return df[['doc_id', 'label', 'title', 'text', 'content', 'model_text']], report


def load_welfake_external(use_cache: bool = True) -> pd.DataFrame:
    """读取"WELFake 去掉 ISOT"的外部测试集（第一次运行约需 3–4 分钟，之后读缓存）。
    Load "WELFake minus ISOT" (first run ~3–4 minutes, cached afterwards).

    缓存与 src/data.py、src/external.py 和原始 WELFake 文件绑定，任何一个变化都会自动重建。
    The cache is keyed on src/data.py, src/external.py and the raw WELFake file;
    any change triggers a rebuild.
    """
    fingerprint = _cache_fingerprint()
    if use_cache and WELFAKE_CACHE.exists() and WELFAKE_CACHE_META.exists():
        meta = json.loads(WELFAKE_CACHE_META.read_text())
        if meta.get('fingerprint') == fingerprint:
            return pd.read_csv(WELFAKE_CACHE, keep_default_na=False)

    df, report = build_welfake_external()
    df.to_csv(WELFAKE_CACHE, index=False)
    WELFAKE_CACHE_META.write_text(json.dumps(
        {'fingerprint': fingerprint, 'report': report}, indent=2, default=str))
    return df


def load_welfake_report() -> dict:
    """最近一次构建的统计信息 / Report from the last build."""
    return json.loads(WELFAKE_CACHE_META.read_text())['report']


# ─────────────────────────────────────────────────────────────────────────────
# LIAR
# ─────────────────────────────────────────────────────────────────────────────

LIAR_COLUMNS = ['id', 'label6', 'statement', 'subject', 'speaker', 'job_title', 'state',
                'party', 'barely_true_count', 'false_count', 'half_true_count',
                'mostly_true_count', 'pants_on_fire_count', 'context']

# LIAR 有 6 级标签，转成二分类。half-true 归为"真"是文献中最常见的做法（与 Wang 2017 之后
# 多数二分类研究一致），但它本身含糊，所以 load_liar() 默认保留 label6，方便以后改用其他划分。
# LIAR has 6 levels; binarised here. Mapping half-true to "real" is the most
# common choice in the binary-LIAR literature, but it is ambiguous — label6 is
# kept so other mappings can be tried later.
LIAR_BINARY = {'true': 1, 'mostly-true': 1, 'half-true': 1,
               'barely-true': 0, 'false': 0, 'pants-fire': 0}


def load_liar() -> pd.DataFrame:
    """读取 LIAR 的 train / valid / test，返回和 ISOT 相同风格的列。
    Load LIAR train / valid / test with ISOT-style columns.

    LIAR 只有一句陈述，没有标题，所以 title 为空、text 就是 statement。
    LIAR has a single statement and no title: title is '' and text is the statement.
    split 沿用 LIAR 官方划分（valid 改名为 val）。
    The official LIAR split is kept (valid → val).
    """
    path = download(LIAR_URL, LIAR_FILE, LIAR_MD5)
    frames = []
    with zipfile.ZipFile(path) as z:
        for part, split in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
            with z.open(f'{part}.tsv') as f:
                # quoting=3 (QUOTE_NONE)：陈述里有未配对的引号，按普通字符读取
                # quoting=3: statements contain unbalanced quotes; read them literally
                d = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), sep='\t', header=None,
                                names=LIAR_COLUMNS, quoting=3, keep_default_na=False)
            frames.append(d.assign(split=split))
    df = pd.concat(frames, ignore_index=True)

    df['doc_id'] = 'liar_' + df['id'].str.replace('.json', '', regex=False)
    df['label'] = df['label6'].map(LIAR_BINARY)
    if df['label'].isna().any():
        raise ValueError(f"Unknown LIAR labels: {sorted(df.loc[df['label'].isna(), 'label6'].unique())}")
    df['label'] = df['label'].astype(int)
    df['title'] = ''
    df['text'] = df['statement'].map(clean_text)
    df['content'] = df['text']
    df['model_text'] = to_model_text(df['content'])
    return df[['doc_id', 'label', 'label6', 'title', 'text', 'content', 'model_text', 'split']]


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助函数 / Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_key(text: pd.Series) -> pd.Series:
    """原始正文的规范化前 300 个字符，用来匹配同一篇文章的不同副本。
    First 300 normalised characters of a raw body — matches copies of the same article."""
    return to_model_text(text).str[:300]


def _key_set(keys: pd.Series) -> set:
    """把一列指纹转成集合，并 *去掉 NaN*。
    Turn a fingerprint column into a set, DROPPING NaN.

    为什么必须去掉 NaN：pandas 的 Series.isin() 会把 NaN 和 NaN 视为相等。
    "太短、不参与比较"的指纹记为 NaN；如果集合里留着 NaN，所有这类文章都会被误判为重叠。
    Why: pandas' Series.isin() treats NaN as equal to NaN. Fingerprints that are
    too short to compare are NaN, so leaving NaN in the set would wrongly flag
    every such article as an overlap.
    """
    return set(keys.dropna())


def _english_share(model_text: pd.Series) -> pd.Series:
    """每篇文章中常见英文虚词的占比 / Share of common English function words per article."""
    return model_text.str.split().map(
        lambda words: sum(w in ENGLISH_FUNCTION_WORDS for w in words) / max(1, len(words)))


def _fingerprints(title: pd.Series, text: pd.Series, model_text: pd.Series) -> dict:
    """五种指纹：src/data.py 分组用的四种（全文、标题、开头、结尾），再加上正文中间
    25 词（第 50–75 个词，只用于至少 100 词的文章）。外部测试集要求零重叠，所以比 ISOT
    内部分组多用一种。
    Five fingerprints: the four from src/data.py's grouping (exact, title, lead,
    tail) plus a 25-word middle window (words 50–75, bodies of 100+ words). One
    more than ISOT's grouping, because the external set must have zero overlap."""
    n = NEAR_DUP_LEAD_WORDS
    title_norm = to_model_text(title)
    words = to_model_text(text).str.split()
    long_enough = words.str.len() >= n
    return {
        'exact': model_text,
        'title': title_norm.where(title_norm.str.split().str.len() >= NEAR_DUP_MIN_TITLE_WORDS),
        'lead': words.str[:n].str.join(' ').where(long_enough),
        'tail': words.str[-n:].str.join(' ').where(long_enough),
        'middle': words.str[50:50 + n].str.join(' ').where(words.str.len() >= 100),
    }


def _count_by_label(df: pd.DataFrame) -> dict:
    counts = df['label'].value_counts()
    return {'fake': int(counts.get(0, 0)), 'real': int(counts.get(1, 0))}


def _cache_fingerprint() -> str:
    """src/data.py + 本文件 + 原始 WELFake 的哈希 / Hash of both modules and the raw file."""
    h = hashlib.sha256()
    for path in (PROJECT_ROOT / 'src' / 'data.py', Path(__file__)):
        h.update(path.read_bytes())
    h.update(WELFAKE_MD5.encode())
    return h.hexdigest()
