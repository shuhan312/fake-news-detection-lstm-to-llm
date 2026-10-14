"""
泄漏审计工具 / Leakage-audit tools.

核心思路："捷径探针" / Core idea: shortcut probes
    用一个非常简单的模型（TF-IDF + 逻辑回归），只给它看文章的 *一小部分* 或 *与内容无关*
    的信息（只看标题、只看开头 20 个词、只看标点和大小写的"形状"），看它能分得多准。
    如果这么简单的模型只看这些就能接近 100%，说明数据里存在不需要理解内容就能利用的捷径，
    深度模型的高分也就不能证明它理解了内容。
    Give a very simple model (TF-IDF + logistic regression) only a SLIVER of
    each article, or only content-free information (title only, first 20 words,
    punctuation/casing "shape"). If that alone gets close to 100%, the data
    contains shortcuts, and a deep model's high score proves nothing about
    understanding.

重要约定 / Important convention
    探针只在 train 上训练、在 val 上评估，*绝不碰 test*。
    test 集必须留到最终评估时只用一次，否则它就不再是"没见过的数据"。
    Probes fit on train and score on val — NEVER on test. The test set is
    used exactly once, for final evaluation; otherwise it is no longer unseen.
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data import NEAR_DUP_LEAD_WORDS, NEAR_DUP_MIN_TITLE_WORDS, to_model_text

# ─────────────────────────────────────────────────────────────────────────────
# 1. 残留痕迹统计 / Residual artefact rates
# ─────────────────────────────────────────────────────────────────────────────

# (列名, 正则, 说明) —— 每一项都是只和"来源网站"有关、和内容真假无关的格式特征
# (column, regex, note) — each is a source-formatting feature unrelated to truthfulness
ARTEFACT_PATTERNS = {
    'Reuters 字样 / "Reuters"':            ('text',  r'(?i)\breuters\b'),
    '开头署名 / dateline "(Reuters) -"':   ('text',  r'^[^()]{0,100}\((?:Reuters|AP|AFP|UPI)\)'),
    '空括号 / empty "()"':                 ('text',  r'\(\s*\)'),
    '记者署名 / "(Reporting by"':          ('text',  r'(?i)\((?:reporting|editing|writing) by'),
    '图片来源 / "Featured image"':         ('text',  r'(?i)featured image|image via|photo by'),
    'Getty Images':                         ('text',  r'(?i)getty'),
    '转载来源 / "Via:" / "Read more"':     ('text',  r'(?i)\bvia\s*:|\bread more\b'),
    '推文链接 / pic.twitter.com':          ('text',  r'(?i)pic\.twitter\.com'),
    'URL':                                  ('text',  r'(?i)https?://|www\.'),
    '@用户名 / @handle':                   ('text',  r'@\w+'),
    '弯引号 / curly quotes':               ('text',  r'[‘’“”]'),
    '不换行空格 / \\xa0':                  ('text',  '\xa0'),
    '句号后缺空格 / "word.Word"':          ('text',  r'[a-z]\.[A-Z][a-z]'),
    '标题方括号 / title "[...]"':          ('title', r'\['),
    '标题 VIDEO / WATCH':                  ('title', r'(?i)\bvideo\b|\bwatch\b'),
    '标题感叹号 / title "!"':              ('title', r'!'),
    # 下面两项是文风差异，不是格式痕迹，规则清洗无法也不应该删除 —— 这正是需要跨数据集测试的原因
    # The two below are writing STYLE, not formatting; rules cannot (and should
    # not) remove them — which is exactly why cross-dataset testing is needed
    '文风 / "said"':                       ('text',  r'\bsaid\b'),
    '文风 / "U.S."':                       ('text',  r'U\.S\.'),
}


def artefact_rates(df: pd.DataFrame) -> pd.DataFrame:
    """每种痕迹在假新闻 / 真新闻中出现的比例（%）。
    Share (%) of fake / real articles containing each artefact."""
    rows = {}
    for name, (col, pattern) in ARTEFACT_PATTERNS.items():
        hit = df[col].str.contains(pattern, regex=True)
        rows[name] = {'fake %': 100 * hit[df['label'] == 0].mean(),
                      'real %': 100 * hit[df['label'] == 1].mean()}
    return pd.DataFrame(rows).T.round(1)


def all_caps_title_rate(df: pd.DataFrame, threshold: float = 0.3) -> dict:
    """标题中全大写单词占比 ≥ threshold 的文章比例（%）。
    Share (%) of titles where >= threshold of words are ALL-CAPS."""
    def caps_share(title: str) -> float:
        words = title.split()
        return sum(w.isupper() and len(w) > 1 for w in words) / max(1, len(words))
    hit = df['title'].map(caps_share) >= threshold
    return {'fake %': round(100 * hit[df['label'] == 0].mean(), 1),
            'real %': round(100 * hit[df['label'] == 1].mean(), 1)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. 探针"视角"：同一篇文章，只给模型看其中一部分
#    Probe "views": show the model only one slice of each article
# ─────────────────────────────────────────────────────────────────────────────

def _first_words(s: pd.Series, n: int) -> pd.Series:
    return s.str.split().str[:n].str.join(' ')


def _last_words(s: pd.Series, n: int) -> pd.Series:
    return s.str.split().str[-n:].str.join(' ')


def _shape(s: pd.Series) -> pd.Series:
    """把小写字母换成 x、大写字母换成 X、数字换成 9，只保留标点、空格和大小写"形状"。
    Replace lowercase → x, uppercase → X, digits → 9: only punctuation,
    spacing and casing "shape" remain — no word meaning at all."""
    return (s.str.replace(r'[a-z]', 'x', regex=True)
             .str.replace(r'[A-Z]', 'X', regex=True)
             .str.replace(r'[0-9]', '9', regex=True))


# 每个视角：(取文本的函数, analyzer, ngram_range)
# 词级视角用 1-2 gram；形状视角用 1-3 个字符的 n-gram（形状没有"词"的概念）
# Each view: (text function, analyzer, ngram_range). Word views use 1-2 grams;
# shape views use 1-3 character n-grams (shapes have no "words").
PROBE_VIEWS = {
    '完整模型输入 / full model_text':       (lambda d: d['model_text'], 'word', (1, 2)),
    '只看标题 / title only':                (lambda d: to_model_text(d['title']), 'word', (1, 2)),
    '只看正文前 20 词 / body first 20':     (lambda d: _first_words(to_model_text(d['text']), 20), 'word', (1, 2)),
    '只看正文后 20 词 / body last 20':      (lambda d: _last_words(to_model_text(d['text']), 20), 'word', (1, 2)),
    '形状（保留大小写标点）/ shape of content': (lambda d: _shape(d['content']), 'char', (1, 3)),
    '形状（模型输入）/ shape of model_text':  (lambda d: _shape(d['model_text']), 'char', (1, 3)),
}


def run_probe(train: pd.DataFrame, evaluate: pd.DataFrame, text_fn,
              analyzer: str = 'word', ngram_range=(1, 2)):
    """在 train 上训练 TF-IDF + 逻辑回归，在 evaluate 上评估。
    Fit TF-IDF + logistic regression on `train`, score on `evaluate`.

    Returns:
        (metrics dict, fitted vectorizer, fitted classifier)
    """
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                 max_features=50_000, min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(text_fn(train))   # 词表只从训练集建立 / vocab from train only
    x_eval = vectorizer.transform(text_fn(evaluate))

    clf = LogisticRegression(C=4.0, max_iter=2000)
    clf.fit(x_train, train['label'])
    prob = clf.predict_proba(x_eval)[:, 1]

    metrics = {'accuracy %': round(100 * accuracy_score(evaluate['label'], prob > 0.5), 1),
               'AUC': round(roc_auc_score(evaluate['label'], prob), 3)}
    return metrics, vectorizer, clf


def run_probe_suite(train: pd.DataFrame, evaluate: pd.DataFrame,
                    views: dict = PROBE_VIEWS) -> pd.DataFrame:
    """依次运行所有探针视角，返回结果表。
    Run every probe view and return a results table."""
    results = {}
    for name, (text_fn, analyzer, ngram_range) in views.items():
        metrics, _, _ = run_probe(train, evaluate, text_fn, analyzer, ngram_range)
        results[name] = metrics
    return pd.DataFrame(results).T


def top_features(vectorizer, clf, k: int = 25) -> pd.DataFrame:
    """逻辑回归权重最大的特征：模型判断"真"或"假"时最依赖哪些词。
    Features with the largest logistic-regression weights: what the model
    relies on most for each class."""
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return pd.DataFrame({'→ real (label 1)': names[order[::-1][:k]],
                         '→ fake (label 0)': names[order[:k]]})


# ─────────────────────────────────────────────────────────────────────────────
# 3. 划分完整性检查 / Split-integrity checks
# ─────────────────────────────────────────────────────────────────────────────

def split_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """检查 val / test 里有多少文章和 train 共享某个"指纹"。
    Count val / test articles that share a fingerprint with any train article.

    - 完全相同、标题相同、开头相同：管线的去重和分组直接处理了这几项，应该为 0。
    - 结尾相同：管线也用它分组，但被超过 NEAR_DUP_MAX_TAIL_GROUP 篇共用的结尾
      （固定页脚）会被跳过，所以这里可能不为 0，需要人工看一下是不是页脚。
    - 中间相同：管线 *没有* 用它分组，是独立的交叉检验，看看还有没有漏网的近似重复。
    - exact / title / lead are handled by the pipeline and must be 0.
    - tail is used too, except tails shared by > NEAR_DUP_MAX_TAIL_GROUP docs
      (boilerplate), so a small non-zero count is possible — inspect it.
    - middle is NOT used by the pipeline: an independent cross-check.
    """
    title = to_model_text(df['title'])
    body_words = to_model_text(df['text']).str.split()
    n = NEAR_DUP_LEAD_WORDS
    long_enough = body_words.str.len() >= n
    keys = {
        '完全相同 / exact model_text': df['model_text'],
        '标题相同 / same title':
            title.where(title.str.split().str.len() >= NEAR_DUP_MIN_TITLE_WORDS),
        f'开头 {n} 词相同 / same lead': body_words.str[:n].str.join(' ').where(long_enough),
        f'结尾 {n} 词相同 / same tail': body_words.str[-n:].str.join(' ').where(long_enough),
        # 第 50–75 个词：只检查至少 100 词的文章，避免和开头 / 结尾重叠
        # words 50–75, only for bodies of 100+ words so it never overlaps lead / tail
        f'中间 {n} 词相同 / same middle (independent check)':
            body_words.str[50:50 + n].str.join(' ').where(body_words.str.len() >= 100),
    }
    is_train = df['split'] == 'train'
    rows = {}
    for name, key in keys.items():
        train_keys = set(key[is_train].dropna())
        rows[name] = {s: int(key[(df['split'] == s)].dropna().isin(train_keys).sum())
                      for s in ('val', 'test')}
    return pd.DataFrame(rows).T


def groups_cross_splits(df: pd.DataFrame) -> int:
    """跨越多个集合的近似重复组数量，应该为 0。
    Number of near-duplicate groups spanning more than one split — must be 0."""
    return int((df.groupby('group_id')['split'].nunique() > 1).sum())


def make_raw_view(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """构造"清洗前"的对照数据：同一批文章、同一个划分，只是文本换回原始版本。
    Build the "before cleaning" control: same articles, same split, raw text.

    这样"清洗前 vs 清洗后"的差异只来自清洗本身，而不是来自样本或划分不同。
    Before/after differences then come from cleaning alone, not from
    different articles or a different split.
    """
    out = cleaned[['doc_id', 'label', 'split']].merge(
        raw[['doc_id', 'title', 'text']], on='doc_id', how='left')
    out['content'] = out['title'] + ' ' + out['text']
    out['model_text'] = to_model_text(out['content'])
    return out
