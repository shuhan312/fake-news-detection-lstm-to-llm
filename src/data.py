"""
统一数据管线 / Unified data pipeline.

所有 notebook（LSTM、BiLSTM、DistilBERT、LLM）都应该通过 load_dataset() 拿数据，
这样每个模型看到的是 *完全相同* 的清洗结果和 *完全相同* 的 train / val / test 划分，
模型之间的对比才公平。
Every notebook (LSTM, BiLSTM, DistilBERT, LLM) should get its data from
load_dataset(), so every model sees *identical* cleaned text and *identical*
train / val / test splits — the only way model comparisons are fair.

管线步骤 / Pipeline steps
    1. load_raw()          读取 Fake.csv / True.csv，给每篇文章一个稳定 ID
                           read the raw CSVs and give each article a stable ID
    2. clean_title() /     对称清洗：真新闻侧（Reuters）和假新闻侧（假新闻网站）
       clean_text()        的来源痕迹都删除，标题和正文都清洗
                           symmetric cleaning: strip source artefacts of BOTH
                           classes, in BOTH title and body
    3. 长度过滤            正文少于 MIN_BODY_WORDS 个词的文章删除
       length filter       drop articles whose body is too short
    4. 精确去重            规范化后完全相同的文章只保留一篇
       exact de-dup        keep one copy of normalised-identical articles
    5. 近似重复分组        标题相同、正文开头 25 词相同或结尾 25 词相同的文章归为同一组
       near-dup grouping   articles sharing a title, a 25-word lead or a 25-word
                           tail form a group
    6. 按组分层划分        同一组的文章只会进入同一个集合，避免"换个版本"的泄漏
       grouped split       a group never straddles train / val / test

为什么不直接删掉近似重复？/ Why group near-duplicates instead of deleting them?
    同一事件的不同版本（例如 Reuters 更新稿）本身是合法数据，删掉会浪费样本；
    真正的问题只是它们不能分散在训练集和测试集两边。按组划分正好解决这一点。
    Different versions of the same story are legitimate data; the only problem
    is when they straddle train and test. Grouped splitting fixes exactly that.

缓存 / Caching
    完整清洗大约需要 2 分钟，所以结果缓存在 data/processed/clean_dataset.csv。
    缓存与"本文件的内容 + 原始数据的哈希"绑定：改了清洗规则或原始数据，缓存自动失效。
    A full build takes ~2 minutes, so results are cached. The cache is keyed on
    a hash of THIS file plus the raw data, so editing the rules or the raw CSVs
    invalidates it automatically.

划分锁定 / Split locking
    data/processed/splits.csv 记录每篇文章属于哪个集合，并提交到 git。
    重新构建时如果划分和已保存的不一致，会直接报错，防止不同实验悄悄用了不同的测试集。
    splits.csv records each article's split and is committed to git. If a
    rebuild produces a different split, load_dataset() raises instead of
    silently giving later experiments a different test set.
"""

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ─────────────────────────────────────────────────────────────────────────────
# 路径与参数 / Paths and parameters
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
CACHE_PATH = PROCESSED_DIR / 'clean_dataset.csv'
CACHE_META_PATH = PROCESSED_DIR / 'clean_dataset_meta.json'
SPLITS_PATH = PROCESSED_DIR / 'splits.csv'

SEED = 42
MIN_BODY_WORDS = 20            # 正文最少词数 / minimum body length in words
NEAR_DUP_LEAD_WORDS = 25       # 用正文前 N 个词判断近似重复 / lead length for near-dup key
NEAR_DUP_MIN_TITLE_WORDS = 5   # 太短的标题（如 "Watch"）不参与分组，避免误合并
                               # very short titles are not used as keys (too generic)
NEAR_DUP_MAX_TAIL_GROUP = 25   # 同一结尾被超过 25 篇共用时，多半是没清掉的固定页脚，不是重复文章。
                               # 清洗后最大的真实重复是 12 篇 Factbox 版本，所以 25 留有余量。
                               # A tail shared by >25 docs is leftover boilerplate, not a
                               # duplicate. The largest genuine one after cleaning is 12.
TEST_FRACTION_FOLDS = 5        # 5 折取 1 折 → 20% 测试集 / 1 of 5 folds → 20% test
VAL_FRACTION_FOLDS = 5         # 剩余部分再 5 折取 1 → 16% 验证集，64% 训练集
                               # 1 of 5 of the rest → 16% val, 64% train

OUTPUT_COLUMNS = ['doc_id', 'label', 'title', 'text', 'content', 'model_text',
                  'group_id', 'split']


# ─────────────────────────────────────────────────────────────────────────────
# 1. 读取原始数据 / Load raw data
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """读取原始 CSV，返回带 doc_id 和 label 的 DataFrame（未清洗）。
    Read the raw CSVs and return an uncleaned DataFrame with doc_id and label.

    doc_id 形如 'fake_00012' / 'true_00345'，即"来源文件 + 原始行号"。
    它不依赖任何随机打乱，所以永远稳定，可以用来跨 notebook 对齐同一篇文章。
    doc_id is "source file + original row number" — independent of any
    shuffling, so it is stable and can align articles across notebooks.

    subject 和 date 列保留在这里只供审计使用；clean 后的数据集不包含它们。
    subject / date are kept here for auditing only; the cleaned dataset drops them.
    """
    # keep_default_na=False：防止标题恰好是 "NA" / "null" 之类的字符串被读成缺失值
    # keep_default_na=False stops titles like "NA" / "null" turning into NaN
    fake = pd.read_csv(raw_dir / 'Fake.csv', keep_default_na=False)
    true = pd.read_csv(raw_dir / 'True.csv', keep_default_na=False)

    fake['label'] = 0
    true['label'] = 1
    fake['doc_id'] = [f'fake_{i:05d}' for i in range(len(fake))]
    true['doc_id'] = [f'true_{i:05d}' for i in range(len(true))]

    df = pd.concat([fake, true], ignore_index=True)
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str)
    return df[['doc_id', 'label', 'title', 'text', 'subject', 'date']]


# ─────────────────────────────────────────────────────────────────────────────
# 2. 清洗规则 / Cleaning rules
#
# 每条规则都来自对原始数据的实际检查（比例见 notebooks/04_leakage_audit.ipynb）。
# 原则：只删"来源格式痕迹"，不删新闻内容本身；规则对两类文章同时生效。
# Every rule comes from inspecting the raw data (rates in 04_leakage_audit).
# Principle: remove source FORMATTING artefacts, never news content; every rule
# is applied to both classes.
#
# 性能提示：所有"在结尾查找"的规则都只在最后 N 个字符里搜索，
# 避免 `.{0,n}...$` 这类在长文本上会严重回溯的写法。
# Performance: "at the end" rules only search the last N characters instead of
# `.{0,n}...$` patterns, which backtrack badly on long texts.
# ─────────────────────────────────────────────────────────────────────────────

# 弯引号 / 长破折号 → ASCII。真新闻正文 50% 含弯引号，假新闻 0%，是纯格式差异。
# Curly quotes / long dashes → ASCII. 50% of real bodies vs 0% of fake ones
# contain curly quotes: a pure formatting difference.
_CHAR_MAP = str.maketrans({
    '‘': "'", '’': "'", '‚': "'", '‛': "'", '′': "'",
    '“': '"', '”': '"', '„': '"', '‟': '"', '″': '"',
    '–': '-', '—': '-', '―': '-', '−': '-',
})

_MONTHS = ('January|February|March|April|May|June|July|August|September|'
           'October|November|December')

# ── 真新闻（Reuters）侧 / Real-news (Reuters) side ────────────────────────────

# 开头的更正说明，例如 "(In Nov 15 item, corrects third paragraph ...)"
# Leading correction notes, e.g. "(In Nov 15 item, corrects third paragraph ...)"
_RE_LEADING_NOTE = re.compile(
    r'^\s*\((?=[^()]*\b(?:corrects?|corrected|refile[sd]?|clarif\w*|'
    r'this story|this version)\b)[^()]{0,400}\)\s*', re.I)

# 开头的地名署名，例如 "WASHINGTON (Reuters) - "、"SEVNICA, Slovenia (Reuters) - "、
# 以及什么都不带的 "(Reuters) - "。旧版正则 [\w\s]+\(Reuters\) 漏掉了后两种。
# 用 ^ 锚定在开头，并限制在 100 个字符内，所以不会误删正文中间的句子。
# Leading datelines incl. "SEVNICA, Slovenia (Reuters) - " and a bare
# "(Reuters) - " — both missed by the old [\w\s]+\(Reuters\) regex. Anchored
# with ^ and capped at 100 chars, so it cannot eat a sentence mid-body.
_RE_DATELINE = re.compile(r'^[^()\n]{0,100}?\((?:Reuters|AP|AFP|UPI)\)\s*-?\s*')

# 正文任意位置的更正 / 重发说明 / Correction or refile notes anywhere
_RE_NOTE_ANYWHERE = re.compile(
    r'\((?=[^()]*\b(?:corrects?|corrected|refile[sd]?|this story|this version)\b)'
    r'[^()]{0,400}\)', re.I)

# 结尾的署名，例如 "(Reporting by ...; Editing by ...)"
# Reuters credit lines, e.g. "(Reporting by ...; Editing by ...)"
_RE_REUTERS_CREDITS = re.compile(
    r'\((?=[^()]*\b(?:reporting|editing|writing|compiled|additional reporting)'
    r'\s+by\b)[^()]{0,600}\)', re.I)

# Reuters 稿件编号 "[L1N1OD2AC]"、推文时间戳 "[0746 EST]" / "[1606 EDT]"，
# 以及推文合集里的 "Source link:" 标记
# Reuters story codes "[L1N1OD2AC]", tweet timestamps "[0746 EST]" / "[1606 EDT]"
# and the "Source link:" marker used in the tweet round-ups
_RE_STORY_CODE = re.compile(r'\[[A-Z0-9]{6,}\]|\[\d{3,4} E[SD]T\]|(?i:\bsource link\s*:?)')

# 95 篇真新闻共用的"特朗普推文合集"固定开头 / Fixed header of 95 "Trump tweets" articles
_RE_TRUMP_TEMPLATE = re.compile(
    r'The following statements\s+were posted to the verified Twitter accounts'
    r'.{0,250}?confirmed their accuracy\.?', re.S)

# 路透社的赞助声明（共 330 篇真新闻）和 "^^^^ Tales from the Trail" 链接区块
# Reuters sponsorship disclaimers (330 real articles) and the "^^^^ Tales from
# the Trail" link block
_RE_REUTERS_BOILERPLATE = re.compile(
    r'This article was funded in part by SAP.{0,250}?creation or production\.?'
    r'|SAP is the sponsor of this (?:coverage|content).{0,250}?'
    r'(?:News Agency|role in this coverage)\.?'
    r"|(?:Follow Trump's impact on|See how the administration's actions affect)"
    r'.{0,150}?at The Trump Effect')
# 括号里的 "点击这里" 导航语，例如 "(For a graphic on DACA, click here)"，两侧都有
# Bracketed navigation hints such as "(For a graphic on DACA, click here)" — both sides
_RE_CLICK_HERE_PAREN = re.compile(r'\((?=[^()]*\bclick here\b)[^()]{0,200}\)', re.I)
_RE_REUTERS_LINK_BLOCK = re.compile(r'\^{5,}|\bTales from the Trail\b')

# 通讯社名称：Reuters 在真新闻中几乎篇篇出现，是最直接的标签泄漏
# Agency names: "Reuters" appears in almost every real article — direct leakage
_RE_AGENCY = re.compile(r'\((?:Reuters|AP|AFP|UPI)\)|\bReuters\b')

# ── 假新闻网站侧 / Fake-site side ─────────────────────────────────────────────

# 网页嵌入代码：Facebook 的 JavaScript（159 篇假新闻）、HTML 标签，
# 以及 Facebook 嵌入帖子的说明 "Posted by X on Tuesday, April 5, 2016"
# Embed code: Facebook JavaScript (159 fake articles), HTML tags, and the
# Facebook embed caption "Posted by X on Tuesday, April 5, 2016"
_RE_EMBED_CODE = re.compile(
    r'\(function\(d, s, id\).{0,800}?\}\(document[^)]{0,60}\)\);?'
    r'|//\s*<\s*!\[CDATA\[|//\s*\]\]>'
    r'|<[^<>]{0,500}>'
    r'|Posted by [^.]{1,80}? on (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),'
    r' (?:' + _MONTHS + r') \d{1,2}, \d{4}', re.S)

# 假新闻网站反复出现的固定段落：作者简介、电台节目归档、视频嵌入说明
# Recurring fake-site boilerplate: author bio, radio-show archive lines,
# video-embed captions
_RE_FAKE_BOILERPLATE = re.compile(
    r'(?i)focusing on film, philosophy, geopolitics.{0,300}?mass media matrix\.?'
    r'|download episode #?\s*\d+|sunday wire radio show archives'
    r'|\*?note: this episode may contain strong language and mature themes\*?'
    r'|the daily show with trevor noah get more:\s*the daily show full episodes,\s*'
    r'the daily show on facebook,\s*the daily show video archive')

# 图片来源说明，例如 "Featured image via Getty Images"（35.7% 的假新闻，真新闻 0%）
# Image credits, e.g. "Featured image via Getty Images" (35.7% fake vs 0% real)
_CREDIT = (r'(?:featured image|feature image|header image|top image|'
           r'images? (?:via|by|credit|courtesy)|photos? (?:via|by|credit|courtesy)|'
           r'screen ?(?:shot|grab|capture) via)')
_RE_CREDIT = re.compile(_CREDIT, re.I)
# 正文中间的图片说明：只删说明短语 + 后面最多 4 个大写开头的词（人名 / 网站名）+ 可选的
# "/Getty Images"，不会截掉后面的正文。
# 遇到常见句首词（He、The、This……）就停下，因为那是下一句正文的开头，
# 例如 "Image via Facebook He killed..." 只删 "Image via Facebook"。
# Mid-body credits: remove only the phrase + up to 4 capitalised words (a name
# or site) + optional "/Getty Images" — the rest of the body is kept. Stops at
# common sentence starters, so "Image via Facebook He killed..." keeps "He killed...".
_SENTENCE_STARTERS = (r'(?:He|She|It|I|We|You|They|The|This|That|These|Those|A|An|'
                      r'In|On|At|But|And|So|If|His|Her|Their|There|When|What|Why|How)\b')
_RE_CREDIT_INLINE = re.compile(
    r'(?i:' + _CREDIT + r')\s*:?\s*'
    r'(?:(?!' + _SENTENCE_STARTERS + r')[A-Z][\w.&\'-]*\s*(?:/\s*)?){0,4}'
    r'(?:Getty Images)?')
# 括号形式的图片说明，例如 "(Photo by Tasos Katopodis/Getty Images)"
# Parenthesised credits, e.g. "(Photo by Tasos Katopodis/Getty Images)"
_RE_CREDIT_PAREN = re.compile(
    r'\((?=[^()]*\b(?:getty|photo|image|screenshot|screengrab)s?\b)[^()]{0,200}\)', re.I)
_RE_GETTY = re.compile(r'\bGetty Images\b', re.I)

# 结尾的转载来源 / "阅读更多"，例如 "Via: Gateway Pundit"、"Read more: ..."
# 要求 via / source 后面带冒号，避免误删 "sent via email." 这类正常句子。
# Trailing source / read-more lines, e.g. "Via: Gateway Pundit". via / source
# must be followed by a colon so normal phrases like "sent via email." survive.
# "For entire story: WSJ" 出现在 440 篇假新闻结尾，真新闻 0 篇
# "For entire story: WSJ" ends 440 fake articles and 0 real ones
_RE_TAIL_SOURCE = re.compile(
    r'(?i)\bvia\s*:|\bsource\s*:|\bh/t\b|\bread more\b|\bfor (?:the )?entire story\b')

# 网站名称，例如 "21st Century Wire"、"@21WIRE.TV"。必须在删除 @用户名 *之前* 处理，
# 否则 "@21WIRE" 先被删掉，会残留 ".TV"。
# Site names such as "21st Century Wire" / "@21WIRE.TV". Must run BEFORE handle
# removal, otherwise "@21WIRE" goes first and ".TV" is left behind.
_RE_SITE_NAMES = re.compile(r'(?i)@?\b21wire(?:\.tv)?\b|21st\s+century\s+wire')

# 订阅广告和转载标记 / Subscription plugs and re-post markers
_RE_SITE_SIGNATURES = re.compile(
    r'(?i)support our work by subscribing[^.]{0,120}'
    r'|\(source:[^)]{0,100}\)|\bread more\s*:?|\bvia\s*:|\bsource\s*:|\bh/t\b'
    r'|\bfor (?:the )?entire story\s*,?\s*:?')

# 嵌入推文的签名，例如 "(@realDonaldTrump) December 31, 2017"（24% 假新闻，0.2% 真新闻）
# Embedded-tweet signatures, e.g. "(@realDonaldTrump) December 31, 2017"
_RE_TWEET_SIG = re.compile(r'\(@\w+\)\s*(?:' + _MONTHS + r')\s+\d{1,2},\s*\d{4}')

# 链接和域名（旧版只删 http/www，漏掉了 pic.twitter.com 和 "Rushlimbaugh.com" 这类）
# URLs and bare domains (the old rule missed pic.twitter.com, "Rushlimbaugh.com")
# 域名后缀只匹配小写：假新闻常见 "the state.Gov. Cuomo" 这种句号后缺空格的情况，
# 忽略大小写会把它误当成域名删掉，而且只伤到假新闻一侧。
# TLDs match lower-case only: case-insensitive matching would treat the fake
# side's missing-space "the state.Gov. Cuomo" as a domain and delete it.
# 链接路径不包含右括号，所以 "(bit.ly/2xYz)" 删除后括号能成对清理掉，不会残留 "("。
# URL paths stop at a closing bracket, so "(bit.ly/2xYz)" leaves a matched,
# removable "()" instead of a dangling "(".
_URL_PATH = r'[^\s)\]]'
_RE_URL = re.compile(
    r'(?i:https?://' + _URL_PATH + r'+|www\.' + _URL_PATH + r'+|pic\.twitter\.com/'
    + _URL_PATH + r'+)'
    r'|\b[\w-]+\.(?:com|org|net|gov|tv|co\.uk|ly)\b(?:/' + _URL_PATH + r'*)?')
_RE_HANDLE = re.compile(r'@\w+')

# ── 通用格式修复 / Generic formatting fixes ───────────────────────────────────

# 假新闻爬取时丢了换行，导致 "safety.Featured" 这种句号后缺空格（86.7% vs 0.5%）
# Fake articles lost their line breaks when scraped: "safety.Featured" (86.7% vs 0.5%)
_RE_MISSING_SPACE_PUNCT = re.compile(r'(?<=[a-z0-9)][.!?:;,])(?=[A-Z])')
# "2017Featured" → "2017 Featured"
_RE_MISSING_SPACE_DIGIT = re.compile(r'(?<=\d)(?=[A-Z][a-z])')
# "FacebookHe" → "Facebook He"。要求大写字母前面至少有 2 个小写字母，
# 所以 "McDonald"、"DeSantis"、"iPhone" 不会被拆开。
# 代价是 "YouTube" 会变成 "You Tube"，但这条规则对两类文章同时生效，不会引入新的偏差。
# Requires 2+ lowercase letters before the capital, so "McDonald", "DeSantis"
# and "iPhone" stay intact. "YouTube" becomes "You Tube", but the rule applies
# to both classes equally, so it adds no bias.
_RE_MISSING_SPACE_CAMEL = re.compile(r'(?<=[a-z]{2})(?=[A-Z][a-z])')

# 删除上面内容后留下的空括号，例如旧管线里的 "() - Puerto Rico"
# Empty brackets left behind, e.g. the old pipeline's "() - Puerto Rico"
_RE_EMPTY_BRACKETS = re.compile(r'\(\s*[-,;:./]*\s*\)|\[\s*[-,;:./]*\s*\]')
_RE_LEADING_JUNK = re.compile(r'^[\s\-:;,.]+')
_RE_WHITESPACE = re.compile(r'\s+')

# ── 标题专用 / Title-only rules ───────────────────────────────────────────────

# 方括号标签，例如 "[VIDEO]"（假新闻标题 20.8%，真新闻 0%）
# Square-bracket tags, e.g. "[VIDEO]" (20.8% fake titles vs 0% real)
_RE_TITLE_SQUARE_TAG = re.compile(r'\[[^\]]{0,40}\]')
# 圆括号标签，例如 "(VIDEO)"、"(TWEETS)"、"(IMAGES)"
# Round-bracket tags, e.g. "(VIDEO)", "(TWEETS)", "(IMAGES)"
_RE_TITLE_ROUND_TAG = re.compile(
    r'\((?=[^()]*\b(?:video|tweet|image|photo|pic|audio|watch|details|graphic|'
    r'screenshot|breaking)s?\b)[^()]{0,40}\)', re.I)
# 开头的吸睛前缀，例如 "WATCH: "、"BREAKING: "
# Click-bait prefixes, e.g. "WATCH: ", "BREAKING: "
_RE_TITLE_PREFIX = re.compile(
    r'^\s*(?:watch|breaking|just in|boom|wow|must see|video|lol|omg)\b\s*[:!\-]*\s*', re.I)
# 带冒号的栏目 / 体裁标签。真新闻侧："Factbox:"（397 篇）、"Exclusive:"（200 篇）等；
# 假新闻侧："REPORT:"、"FLASHBACK:"、"SUNDAY SCREENING:" 等。
# 注意："BOMBSHELL:"、"BUSTED:" 这类 *煽动性用词* 故意保留：它们是措辞风格，
# 是模型应该被允许学习的语言特征，而不是网站的排版格式。
# Section / genre labels followed by a colon — real: "Factbox:" (397),
# "Exclusive:" (200)…; fake: "REPORT:", "FLASHBACK:", "SUNDAY SCREENING:"…
# Sensational words such as "BOMBSHELL:" / "BUSTED:" are deliberately KEPT:
# they are word choice the model may legitimately learn, not site formatting.
_RE_TITLE_SECTION_TAG = re.compile(
    r'^\s*(?:factbox|exclusive|highlights|timeline|explainer|analysis|instant view|'
    r'breakingviews|report|news|update|flashback|sunday screening)\s*:\s*', re.I)
# 路透社推文汇总的标题模板 "Trump on Twitter (Dec 28) - Global Warming"（只在真新闻中出现）
# Reuters tweet round-up template "Trump on Twitter (Dec 28) - ..." (real only)
_RE_TITLE_TWITTER_TEMPLATE = re.compile(r'^\s*Trump on Twitter \([^)]{0,20}\)\s*-?\s*')


def _normalise_unicode(s: str) -> str:
    """HTML 实体解码 → 弯引号转直引号 → 去重音 → 只保留 ASCII。
    Decode HTML entities → straighten quotes → strip accents → ASCII only.

    旧版直接删除非 ASCII 字符，会把 "Rosselló" 变成 "Rossell"；
    这里先用 NFKD 分解再去掉重音符号，得到 "Rossello"。
    The old rule deleted non-ASCII chars ("Rosselló" → "Rossell"); NFKD
    decomposition keeps the base letter ("Rossello"). It also turns the
    non-breaking space (\\xa0, 5% of real articles only) into a normal space.
    """
    s = html.unescape(s)
    s = s.translate(_CHAR_MAP)
    s = unicodedata.normalize('NFKD', s)
    return s.encode('ascii', 'ignore').decode('ascii')


def _cut_tail(s: str, pattern: re.Pattern, window: int, max_tail_words: int) -> str:
    """如果 pattern 出现在结尾附近，并且从它开始到结尾不超过 max_tail_words 个词，
    就把这一段删掉。
    If `pattern` occurs near the end AND at most `max_tail_words` words follow
    from there to the end, delete that stretch.

    词数限制保证删掉的只是结尾的说明行（如 "Featured image via X/Getty Images"），
    不会把说明后面的正文也一起截掉。
    The word limit guarantees only a trailing credit line is removed, never
    body text that happens to follow a mid-body credit.
    """
    for m in pattern.finditer(s, max(0, len(s) - window)):
        if len(s[m.start():].split()) <= max_tail_words:
            # 同时删掉说明前面紧挨着的 "("，例如 "(Photo by X/Getty Images)"
            # Also drop an opening "(" directly before the credit
            return s[:m.start()].rstrip(' (')
    return s


def _finish(s: str) -> str:
    """删除空括号、开头残留的标点，并合并多余空白。
    Remove empty brackets and leading punctuation, collapse whitespace."""
    s = _RE_EMPTY_BRACKETS.sub(' ', s)
    s = _RE_WHITESPACE.sub(' ', s).strip()
    return _RE_LEADING_JUNK.sub('', s)


def clean_text(s: str) -> str:
    """清洗文章正文，规则对真假新闻同时生效。
    Clean an article body; every rule applies to both classes.

    顺序很重要 / Order matters:
      - 嵌入代码和固定段落最先删除，因为其他规则会把它们拆碎
        embed code and boilerplate first, before other rules fragment them
      - 开头署名在其他正文替换之前删除，否则 ^ 锚点可能失效
        datelines before other body edits can shift the ^ anchor
      - 先删 URL，再修复缺失空格，否则 URL 里的大小写会被拆开
        URLs before the missing-space fixes, which would split camel-case URLs
      - 先修复缺失空格，再删结尾的图片说明，这样 "safety.Featured image" 能被正确识别
        missing-space fixes before tail cuts, so "safety.Featured image" is found
    """
    s = _normalise_unicode(s)

    # 网页嵌入代码和固定段落：最先删除，因为它们内部包含链接、@ 和日期，
    # 如果先被其他规则拆碎，就无法整体匹配了
    # Embed code and boilerplate first: they contain URLs, "@" and dates that
    # other rules would otherwise break into unmatchable fragments
    s = _RE_EMBED_CODE.sub(' ', s)
    s = _RE_REUTERS_BOILERPLATE.sub(' ', s)
    s = _RE_FAKE_BOILERPLATE.sub(' ', s)

    # 真新闻侧 / real-news side
    s = _RE_TRUMP_TEMPLATE.sub(' ', s)
    s = _RE_LEADING_NOTE.sub('', s)
    s = _RE_DATELINE.sub('', s)
    s = _RE_NOTE_ANYWHERE.sub(' ', s)
    s = _RE_REUTERS_CREDITS.sub(' ', s)
    s = _RE_CLICK_HERE_PAREN.sub(' ', s)
    s = _RE_STORY_CODE.sub(' ', s)

    # 假新闻侧：推文签名 → 链接 → @用户名
    # 删 @用户名 之后会留下空括号，由 _finish() 统一清理，避免旧版残留 "()" 的问题。
    # Fake side: tweet signatures → URLs → handles. Empty brackets left behind are
    # removed in _finish(), fixing the old pipeline's leftover "()" artefact.
    s = _RE_TWEET_SIG.sub(' ', s)
    s = _RE_SITE_NAMES.sub(' ', s)
    s = _RE_URL.sub(' ', s)
    s = _RE_HANDLE.sub(' ', s)

    # 修复爬取时丢失的空格 / restore spaces lost during scraping
    s = _RE_MISSING_SPACE_PUNCT.sub(' ', s)
    s = _RE_MISSING_SPACE_DIGIT.sub(' ', s)
    s = _RE_MISSING_SPACE_CAMEL.sub(' ', s)

    # 结尾的图片说明和转载来源可能叠在一起（例如 "Via: X Featured image via Y"），
    # 所以重复截几次，直到不再变化。
    # Trailing credits and source lines can be stacked, so cut repeatedly.
    for _ in range(3):
        before = s
        s = s.rstrip()
        s = _cut_tail(s, _RE_CREDIT, window=300, max_tail_words=15)
        s = _cut_tail(s, _RE_TAIL_SOURCE, window=200, max_tail_words=25)
        s = _cut_tail(s, _RE_REUTERS_LINK_BLOCK, window=800, max_tail_words=90)
        if s == before:
            break

    # 正文中间的同类痕迹：只删短语本身，保留其余正文
    # Same artefacts mid-body: remove just the phrase, keep the rest of the body
    s = _RE_CREDIT_PAREN.sub(' ', s)
    s = _RE_CREDIT_INLINE.sub(' ', s)
    s = _RE_GETTY.sub(' ', s)
    s = _RE_SITE_SIGNATURES.sub(' ', s)
    s = _RE_AGENCY.sub(' ', s)

    return _finish(s)


def clean_title(s: str) -> str:
    """清洗标题。旧版完全没有清洗标题，而只看标题就能达到约 95% 的准确率。
    Clean a headline. The old pipeline never cleaned titles, which alone gave
    ~95% accuracy.

    大小写不在这里处理：model_text 会统一转为小写，
    所以假新闻标题的全大写写法不会进入模型输入。
    Casing is not changed here: model_text lowercases everything, so all-caps
    fake headlines never reach the models.
    """
    s = _normalise_unicode(s)
    s = _RE_TITLE_SQUARE_TAG.sub(' ', s)
    s = _RE_TITLE_ROUND_TAG.sub(' ', s)
    s = _RE_URL.sub(' ', s)
    s = _RE_HANDLE.sub(' ', s)
    s = _RE_AGENCY.sub(' ', s)
    s = _finish(s)
    # 前缀可能叠加（例如 "BREAKING: WATCH: ..."），所以重复删除
    # Prefixes can stack ("BREAKING: WATCH: ..."), so strip repeatedly
    for _ in range(3):
        stripped = _RE_TITLE_TWITTER_TEMPLATE.sub('', s)
        stripped = _RE_TITLE_SECTION_TAG.sub('', stripped)
        stripped = _RE_TITLE_PREFIX.sub('', stripped)
        if stripped == s:
            break
        s = stripped
    return _finish(s)


def to_model_text(s: pd.Series) -> pd.Series:
    """模型实际读取的文本：全部小写，只保留字母和数字。
    The text the models actually read: lower-case letters and digits only.

    为什么连标点也去掉？/ Why drop punctuation too?
      把所有字母换成 x、只保留标点和大小写"形状"的探针，在旧版（02 notebook）
      清洗后的数据上有 99.5% 的准确率，在本管线清洗后的文本上仍有 95.5%
      （见 04_leakage_audit）。标点和大小写在这个数据集里主要反映来源网站的排版习惯，
      而不是内容真假。Keras Tokenizer 本来就会去标点、转小写；在这里统一处理，
      DistilBERT 也会读到同样的信息，两个模型的对比才公平。
      A shape-only probe (every letter → x) scored 99.5% on the old (02)
      cleaning and still 95.5% on this pipeline's cleaned text (see
      04_leakage_audit): punctuation and casing mostly encode each site's house
      style. Keras' Tokenizer already lower-cases and strips punctuation; doing
      it here gives DistilBERT the same information, keeping comparisons fair.
    """
    # 先把带点的缩写合并："U.S." → "US"、"U.N." → "UN"。
    # Reuters 写 "U.S."，假新闻网站多写 "US"，这只是拼写习惯，不是内容差异。
    # 不合并的话，去标点后 "U.S." 会变成两个单字母，和 "us" 成为两种不同的写法。
    # Collapse dotted acronyms first ("U.S." → "US"): Reuters writes "U.S.",
    # fake sites mostly "US" — a spelling convention, not content.
    s = s.str.replace(r'\b(?:[A-Za-z]\.){2,}',
                      lambda m: m.group(0).replace('.', ''), regex=True)
    return (s.str.lower()
             .str.replace(r'[^a-z0-9]+', ' ', regex=True)
             .str.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 3. 去重与近似重复分组 / De-duplication and near-duplicate grouping
# ─────────────────────────────────────────────────────────────────────────────

def _group_near_duplicates(df: pd.DataFrame) -> np.ndarray:
    """用并查集把"标题相同"、"正文开头相同"或"正文结尾相同"的文章合并成组，返回每行的组号。
    Union-find over shared titles / body leads / body tails; returns a group id per row.

    合并是传递的：A 和 B 标题相同、B 和 C 开头相同，那么 A、B、C 都在同一组。
    Grouping is transitive: if A~B by title and B~C by lead, A, B, C share a group.

    为什么也看结尾？/ Why tails too?
      同一个故事被不同网站转载时，常常换了标题和开头，但结尾的引语不变；
      路透社的更新稿也经常只改开头。
      Re-posts often change the title and lead but keep the closing quote;
      Reuters updates often rewrite only the lead.
    """
    n = len(df)
    parent = np.arange(n)

    def find(i: int) -> int:
        # 路径压缩：查找时把节点直接挂到根节点上 / path compression
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    title_norm = to_model_text(df['title'])
    body_norm = to_model_text(df['text'])
    body_words = body_norm.str.split()

    # 太短的标题和正文不作为分组依据（太泛，会把无关文章错误合并）
    # Short titles / bodies are not used as keys (too generic, would over-merge)
    title_key = title_norm.where(
        title_norm.str.split().str.len() >= NEAR_DUP_MIN_TITLE_WORDS)
    long_enough = body_words.str.len() >= NEAR_DUP_LEAD_WORDS
    lead_key = body_words.str[:NEAR_DUP_LEAD_WORDS].str.join(' ').where(long_enough)
    tail_key = body_words.str[-NEAR_DUP_LEAD_WORDS:].str.join(' ').where(long_enough)

    # (键, 最大组大小)。结尾键加上限：被很多篇共用的结尾是固定页脚，不代表文章重复
    # (key, max group size). Tails are capped: a widely shared tail is boilerplate
    for key, max_size in ((title_key, None), (lead_key, None),
                          (tail_key, NEAR_DUP_MAX_TAIL_GROUP)):
        # groupby 默认跳过 NaN，所以无效的键不会参与合并
        # groupby drops NaN keys, so invalid keys never merge anything
        for positions in key.groupby(key).indices.values():
            if max_size is not None and len(positions) > max_size:
                continue
            root = find(positions[0])
            for p in positions[1:]:
                parent[find(p)] = root

    roots = np.array([find(i) for i in range(n)])
    # 把根节点编号重新映射为 0..k-1 的连续整数，方便阅读
    # Re-number roots as consecutive integers 0..k-1 for readability
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. 划分 / Splitting
# ─────────────────────────────────────────────────────────────────────────────

def _assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """按组分层划分为约 64% / 16% / 20% 的 train / val / test。
    Grouped, stratified split into ~64% / 16% / 20% train / val / test.

    StratifiedGroupKFold 同时保证两件事：
      1. 同一组的文章只出现在一个集合里（防止近似重复泄漏）
      2. 每个集合中真假新闻的比例接近（分层）
    StratifiedGroupKFold guarantees both (1) a group lives in exactly one split
    (no near-duplicate leakage) and (2) similar label ratios in every split.
    """
    split = pd.Series('train', index=df.index)

    outer = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True,
                                 random_state=seed)
    trainval_pos, test_pos = next(outer.split(df, df['label'], df['group_id']))
    split.iloc[test_pos] = 'test'

    trainval = df.iloc[trainval_pos]
    inner = StratifiedGroupKFold(n_splits=VAL_FRACTION_FOLDS, shuffle=True,
                                 random_state=seed)
    _, val_rel = next(inner.split(trainval, trainval['label'], trainval['group_id']))
    split.iloc[trainval_pos[val_rel]] = 'val'
    return split


# ─────────────────────────────────────────────────────────────────────────────
# 5. 构建与加载 / Build and load
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(raw_dir: Path = RAW_DIR, seed: int = SEED):
    """从原始 CSV 完整构建数据集（不读也不写缓存）。
    Build the dataset from the raw CSVs (no caching).

    Returns:
        (df, report)：清洗后的 DataFrame，以及记录每一步删除多少篇文章的字典。
        (df, report): the cleaned DataFrame and a dict counting what each step removed.
    """
    df = load_raw(raw_dir)
    report = {'raw_total': len(df),
              'raw_by_label': _count_by_label(df)}

    # 清洗 / clean
    df['title'] = df['title'].map(clean_title)
    df['text'] = df['text'].map(clean_text)
    df['content'] = (df['title'] + ' ' + df['text']).str.strip()

    # 长度过滤：只看正文，不算标题。约 1,100 篇假新闻只有标题、正文几乎为空，
    # 如果保留下来，"正文很短"本身就会成为一个假新闻信号。
    # Length filter on the BODY only. ~1,100 fake rows are headline-only; keeping
    # them would make "short body" itself a fake-news signal.
    short = df['text'].str.split().str.len() < MIN_BODY_WORDS
    report['removed_short_body_by_label'] = _count_by_label(df[short])
    df = df[~short].reset_index(drop=True)

    # 精确去重：基于模型实际看到的规范化文本
    # Exact de-dup on the normalised text the models actually see
    df['model_text'] = to_model_text(df['content'])

    # 如果同一篇文章在两个类别里都出现，说明标签有冲突，无法判断哪个对，所以整组删除
    # Identical text with conflicting labels: we cannot tell which label is right,
    # so the whole group is dropped
    n_labels = df.groupby('model_text')['label'].transform('nunique')
    conflicting = n_labels > 1
    report['removed_conflicting_label_rows'] = int(conflicting.sum())
    df = df[~conflicting]

    is_dup = df.duplicated(subset='model_text', keep='first')
    report['removed_exact_duplicates_by_label'] = _count_by_label(df[is_dup])
    df = df[~is_dup].reset_index(drop=True)

    # 近似重复分组 / near-duplicate grouping
    df['group_id'] = _group_near_duplicates(df)
    group_sizes = df.groupby('group_id').size()
    group_labels = df.groupby('group_id')['label'].nunique()
    report['near_dup_groups_multi_doc'] = int((group_sizes > 1).sum())
    report['docs_in_multi_doc_groups'] = int(group_sizes[group_sizes > 1].sum())
    report['largest_group_size'] = int(group_sizes.max())
    # 同一组里既有真新闻又有假新闻：通常是假新闻网站转载了 Reuters 的文章，属于标签噪声
    # Mixed-label groups: usually a fake site re-posting a Reuters story (label noise)
    report['mixed_label_groups'] = int((group_labels > 1).sum())

    # 划分 / split
    df['split'] = _assign_splits(df, seed)
    report['final_total'] = len(df)
    report['final_by_split_and_label'] = {
        name: _count_by_label(part) for name, part in df.groupby('split')}

    return df[OUTPUT_COLUMNS], report


def load_dataset(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """所有 notebook 的数据入口：返回清洗、去重、划分好的数据集。
    Single entry point for every notebook: the cleaned, de-duplicated, split data.

    列 / Columns:
        doc_id      稳定 ID，可对应回原始 CSV 的行 / stable ID back to the raw row
        label       0 = fake, 1 = real
        title, text 清洗后的标题和正文（保留大小写和标点，方便人工阅读）
                    cleaned title and body (case and punctuation kept, for reading)
        content     title + ' ' + text
        model_text  模型输入：小写，只含字母和数字 / model input: lower-case alphanumerics
        group_id    近似重复组号 / near-duplicate group id
        split       'train' / 'val' / 'test'

    用法 / Usage:
        df = load_dataset()
        train = df[df.split == 'train']   # 训练 / fit models
        val   = df[df.split == 'val']     # 调参、EarlyStopping / tuning & early stopping
        test  = df[df.split == 'test']    # 只在最终评估时用一次 / final evaluation only
    """
    fingerprint = _fingerprint()

    if use_cache and CACHE_PATH.exists() and CACHE_META_PATH.exists():
        meta = json.loads(CACHE_META_PATH.read_text())
        if meta.get('fingerprint') == fingerprint:
            # keep_default_na=False：空字符串保持为 ''，而不是被读成 NaN
            # keep_default_na=False keeps empty strings as '' instead of NaN
            df = pd.read_csv(CACHE_PATH, keep_default_na=False)
            _check_splits_locked(df)
            if verbose:
                print(f'Loaded cached dataset: {len(df):,} rows ({CACHE_PATH.name})')
            return df

    if verbose:
        print('Building dataset from raw CSVs (this takes ~2 minutes)...')
    df, report = build_dataset()
    _check_splits_locked(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    CACHE_META_PATH.write_text(json.dumps(
        {'fingerprint': fingerprint, 'report': report}, indent=2))
    if verbose:
        print(f'Built and cached dataset: {len(df):,} rows')
    return df


def load_report() -> dict:
    """返回最近一次构建的清洗统计（每一步删除了多少篇文章）。
    Return the cleaning report (how many articles each step removed) from the last build."""
    return json.loads(CACHE_META_PATH.read_text())['report']


def get_splits(df: pd.DataFrame):
    """返回 (train, val, test) 三个 DataFrame / Return (train, val, test) DataFrames."""
    return tuple(df[df['split'] == name].reset_index(drop=True)
                 for name in ('train', 'val', 'test'))


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助函数 / Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_by_label(df: pd.DataFrame) -> dict:
    counts = df['label'].value_counts()
    return {'fake': int(counts.get(0, 0)), 'real': int(counts.get(1, 0))}


def _fingerprint() -> str:
    """本文件内容 + 原始 CSV 内容的哈希；任何一方变化都会让缓存失效。
    Hash of this file and the raw CSVs; any change invalidates the cache."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    for name in ('Fake.csv', 'True.csv'):
        h.update((RAW_DIR / name).read_bytes())
    return h.hexdigest()


def _check_splits_locked(df: pd.DataFrame, path: Path = None) -> None:
    """第一次运行时保存划分；之后每次都检查划分是否和保存的一致。
    Save the split on first run; afterwards verify it never changes silently.

    path 默认是 ISOT 的 data/processed/splits.csv；外部数据集传入自己的划分文件路径。
    path defaults to ISOT's data/processed/splits.csv; external datasets pass their own.
    （默认值在运行时读取 SPLITS_PATH，所以单元测试可以替换它。
    The default is read at call time, so unit tests can monkeypatch SPLITS_PATH.）

    如果你 *有意* 修改了清洗规则并接受新的划分，删除对应的划分文件后重新运行。
    注意：划分改变后，之前所有模型的结果都要用新划分重新跑，否则对比不公平。
    If you INTENTIONALLY changed the cleaning and accept a new split, delete the
    split file and re-run — then re-run every model on the new split.
    """
    path = SPLITS_PATH if path is None else path
    current = df[['doc_id', 'split']].sort_values('doc_id').reset_index(drop=True)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(path, index=False)
        return

    saved = pd.read_csv(path, keep_default_na=False)
    saved = saved.sort_values('doc_id').reset_index(drop=True)
    if not current.equals(saved):
        raise RuntimeError(
            f'The rebuilt train/val/test split differs from {path.name}. '
            f'If the cleaning change was intentional, delete {path.name}, rebuild, and '
            're-run ALL models so they share the new split.')
