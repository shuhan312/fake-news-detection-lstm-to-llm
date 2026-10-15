"""
外部数据处理的单元测试（不需要联网）/ Unit tests for src/external.py (no network needed).

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import external  # noqa: E402


def test_key_set_drops_nan_so_short_keys_never_match():
    # pandas 的 isin() 会把 NaN 和 NaN 视为相等；_key_set 必须去掉 NaN，
    # 否则所有"太短不比较"的文章都会被误判为和 ISOT 重叠
    # pandas isin() treats NaN == NaN; _key_set must drop NaN or every
    # too-short-to-compare article would be flagged as an ISOT overlap
    ext = pd.Series(['shared key', None, 'unique key'])
    isot = pd.Series(['shared key', None])
    assert ext.isin(external._key_set(isot)).tolist() == [True, False, False]


def test_publisher_suffix_removed_but_real_subtitles_kept():
    titles = pd.Series([
        'Fed Officials See Faster Growth - The New York Times',
        "CBS's Dickerson: Lack of Trust in the Media - Breitbart",
        'Boiler Room #62 - Fatal Illusions',            # 真正的副标题，必须保留 / real subtitle
    ])
    out = titles.str.replace(external._RE_PUBLISHER_SUFFIX, '', regex=True).tolist()
    assert out == ['Fed Officials See Faster Growth',
                   "CBS's Dickerson: Lack of Trust in the Media",
                   'Boiler Room #62 - Fatal Illusions']


def test_liar_binary_mapping_covers_all_six_labels():
    assert set(external.LIAR_BINARY) == {'true', 'mostly-true', 'half-true',
                                         'barely-true', 'false', 'pants-fire'}
    assert [external.LIAR_BINARY[k] for k in ('true', 'half-true', 'barely-true', 'pants-fire')] \
        == [1, 1, 0, 0]


def test_english_share_separates_english_from_other_languages():
    texts = pd.Series([
        'the president said on monday that he would not sign the bill',
        'los presidentes de la region se reunieron en la ciudad de mexico',
        'der prasident hat am montag gesagt dass er das gesetz nicht unterschreibt',
    ])
    share = external._english_share(texts)
    assert share[0] >= external.MIN_ENGLISH_SHARE
    assert share[1] < external.MIN_ENGLISH_SHARE
    assert share[2] < external.MIN_ENGLISH_SHARE


def test_fingerprints_include_middle_window_only_for_long_bodies():
    long_body = ' '.join(f'w{i}' for i in range(120))
    short_body = ' '.join(f'w{i}' for i in range(60))
    fp = external._fingerprints(pd.Series(['a five word long title', 'x']),
                                pd.Series([long_body, short_body]),
                                pd.Series(['m1', 'm2']))
    assert fp['middle'][0] == ' '.join(f'w{i}' for i in range(50, 75))
    assert pd.isna(fp['middle'][1])      # 少于 100 词不比较中间段 / <100 words: no middle key
    assert pd.isna(fp['title'][1])       # 少于 5 个词的标题不比较 / <5-word title: no title key


def test_strip_promo_tail_removes_bios_and_plugs_but_keeps_body():
    body = 'The committee voted on Tuesday. Officials said the plan would take effect in May.'
    # Breitbart 作者简介 / Breitbart author bio
    assert external.strip_promo_tail(
        body + ' He is also the political analyst for Armed American Radio. '
               'Follow him on Twitter: . Reach him directly at .') == body
    # 假新闻侧的分享推广 / fake-side share plug
    assert external.strip_promo_tail(
        body + ' Please like and share this story on Facebook or Twitter.') == body
    # 正文结尾提到 Twitter 的普通句子必须保留 / a normal last sentence mentioning Twitter stays
    tweet_end = body + ' He said on Twitter that the vote was rigged.'
    assert external.strip_promo_tail(tweet_end) == tweet_end


def test_source_name_breitbart_removed():
    out = external._RE_SOURCE_NAME.sub('', 'He told Breitbart News Daily and Breitbart Texas that')
    assert 'Breitbart' not in out


def test_promo_tail_survives_split_email_fragment():
    # 原始数据里邮箱被切开（"briefing@nytimes. com"），清洗后结尾会出现 "com." 碎片；
    # 碎片必须并回前一句，否则推广语检查会在 "com." 处停下
    # Split e-mails leave a trailing "com." fragment; it must merge backwards
    body = 'The committee voted on Tuesday. Officials said the plan would take effect in May.'
    assert external.strip_promo_tail(body + ' What do you want to see here? Let us know at . com.') == body


def test_repeated_tail_sentences_are_removed_but_unique_endings_kept():
    bio = 'He was named one of the most influential people in news media in 2016.'
    body = 'Officials said the measure passed {} votes to 40 on Tuesday.'
    texts = pd.Series([body.format(50 + i) + ' ' + bio for i in range(external.BOILERPLATE_MIN_DOCS)]
                      + [body.format(99) + ' The final sentence is unique to this article.'])
    out = external.strip_boilerplate_tails(texts)
    # 在 10 篇文章结尾重复的简介被删除 / the bio repeated across 10 articles is removed
    assert all(bio not in t for t in out[:-1])
    assert out[0] == body.format(50)
    # 只出现一次的结尾保留 / a unique ending is kept
    assert out.iloc[-1].endswith('unique to this article.')


def test_promo_tail_never_removes_long_unpunctuated_body():
    # 没有句号的长段落（例如讽刺新闻的列表）即使包含 "subscribe" 也不能被整段删掉
    # A long unpunctuated block (e.g. a satire list) must not be removed whole,
    # even if it contains a promo word
    body = 'The committee voted on Tuesday. Officials said the plan would take effect in May.'
    long_block = ' '.join(['item'] * 90) + ' subscribe now'
    assert external.strip_promo_tail(body + ' ' + long_block).endswith('subscribe now')
