"""
清洗规则的单元测试 / Unit tests for the cleaning rules in src/data.py.

每个用例都取自原始数据里真实出现过的格式。修改清洗规则之后运行：
Every case mirrors a format seen in the raw data. Run after editing any rule:

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import clean_text, clean_title, to_model_text  # noqa: E402

# 正文中间的说明后面接一段足够长的正文，模拟真实文章（避免被"结尾截断"规则误判）
# Mid-body cases are followed by a realistic amount of body text
LONG_TAIL = (' The story continued for several more paragraphs with quotes from '
             'local officials and residents who described what they had seen that day.')

TEXT_CASES = [
    # ── 真新闻侧 / real-news side ────────────────────────────────────────────
    ('WASHINGTON (Reuters) - U.S. President Donald Trump removed',
     'U.S. President Donald Trump removed'),
    # 旧正则漏掉的情况：开头直接是 "(Reuters)"，以前会残留 "() -"
    # Missed by the old regex: bare "(Reuters)" used to leave "() -"
    ('(Reuters) - Puerto Rico Governor Ricardo Rosselló said',
     'Puerto Rico Governor Ricardo Rossello said'),
    # 地名里带逗号 / dateline containing a comma
    ('SEVNICA, Slovenia (Reuters) - Melania grew up here.',
     'Melania grew up here.'),
    # 开头的更正说明 + 署名 / correction note followed by a dateline
    (' (Corrects paragraph 7 to show Trump issued a warning) HAVANA (Reuters) - Cuba said',
     'Cuba said'),
    # 弯引号 + 结尾的记者署名 / curly quotes + trailing credit line
    ('He said “it’s fine”. (Reporting by Jane Doe; Editing by Bob Roe)',
     'He said "it\'s fine".'),
    ('Temer underwent surgery on Wednesday. [L1N1OD2AC]',
     'Temer underwent surgery on Wednesday.'),
    # 推文合集：EDT 时间戳、"Source link:"、括号里的短链接（不能残留 "("）
    # Tweet round-ups: EDT timestamp, "Source link:", bracketed short link (no "(" left)
    ('HEROES in Texas! [1606 EDT] -- Source link: (bit.ly/2xYz) Next tweet',
     'HEROES in Texas! -- Next tweet'),
    ('The following statements\xa0were posted to the verified Twitter accounts of U.S. '
     'President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are '
     'his own.\xa0Reuters has not edited the statements or confirmed their accuracy.  '
     '@realDonaldTrump : - While the Fake News loves',
     'While the Fake News loves'),
    # 正文中的 "told Reuters"：只删机构名 / "told Reuters" mid-body: drop the name only
    ('and the state.Gov. Cuomo said he told Reuters that McDonald and DeSantis met',
     'and the state. Gov. Cuomo said he told that McDonald and DeSantis met'),

    # ── 假新闻侧 / fake-site side ───────────────────────────────────────────
    ('Rubio should be ashamed of himself.Featured image: Video screen capture',
     'Rubio should be ashamed of himself.'),
    ('Watch Ann make a fool of herself (yet again):Featured image via Stephen Lovekin/Getty Images',
     'Watch Ann make a fool of herself (yet again):'),
    ('will be a great year!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down',
     'will be a great year! Donald J. Trump Trump s tweet went down'),
    ('so little to deliver.  Via: Rushlimbaugh.com',
     'so little to deliver.'),
    # 正文中间的图片说明：只删说明，后面的正文必须保留
    # Mid-body credit: remove the credit, keep the body that follows
    ('I touched that guy s hand.Image via FacebookHe killed his dad.' + LONG_TAIL,
     'I touched that guy s hand. He killed his dad.' + LONG_TAIL),
    ('the people (Photo by Tasos Katopodis/Getty Images)That s the Juggalos.' + LONG_TAIL,
     'the people That s the Juggalos.' + LONG_TAIL),
    # 结尾的括号形式图片说明，不能残留 "(" / trailing parenthesised credit leaves no "("
    ('They looked like kids. (Photo by Drew Angerer/Getty Images)',
     'They looked like kids.'),
    # URL 后面紧贴的单词在原始数据里就粘在一起了，无法恢复，接受丢掉这个词
    # A word glued onto a URL in the raw data cannot be recovered; it is dropped
    ('watch here https://www.youtube.com/watch?v=9_nKOL4h0iUThis woman needs',
     'watch here woman needs'),
    ('21st Century Wire says Ben Stein, reputable professor',
     'says Ben Stein, reputable professor'),
    ('the end. SUPPORT OUR WORK BY SUBSCRIBING & BECOMING A MEMBER @21WIRE.TV',
     'the end.'),

    # 第二轮审计发现的页脚和嵌入代码 / footers and embed code found in audit round 2
    ('He spoke on Monday. (Writing by Paul Taylor) This article was funded in part by SAP. '
     'It was independently created by the Reuters editorial staff. SAP had no editorial '
     'involvement in its creation or production.',
     'He spoke on Monday.'),
    ('Officials met on Monday. SAP is the sponsor of this coverage which is independently '
     'produced by the staff of Reuters News Agency.',
     'Officials met on Monday.'),
    ("Chhabria ruled in favor of Trump. Follow Trump's impact on energy, environment, "
     'healthcare, immigration and the economy at The Trump Effect',
     'Chhabria ruled in favor of Trump.'),
    ('Pena said. (For a graphic on Deferred Action for Childhood Arrivals, click here)',
     'Pena said.'),
    ('in the state legislature. For entire story: WSJ',
     'in the state legislature.'),
    ('got into it!// < ![CDATA[ // < ![CDATA[ (function(d, s, id) { var js, fjs = '
     'd.getElementsByTagName(s)[0]; if (d.getElementById(id)) return; js = d.createElement(s); '
     'js.id = id; js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&#038;version=v2.3"; '
     "fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); "
     '// ]]&gt;Posted by BuzzFeed News on Monday, March 28, 2016Their mother is proud.' + LONG_TAIL,
     'got into it! Their mother is proud.' + LONG_TAIL),
    ('You can watch it here:<iframe width= 560  height= 315  src= https://www.youtube.com/embed/2Rr '
     'frameborder= 0  allowfullscreen></iframe>' + LONG_TAIL,
     'You can watch it here:' + LONG_TAIL),
    ('The vote is in November. ^^^^^^^^^^^^^^^^ U.S. Presidential Election 2016 '
     'cpurl://apps.cp./cms/?pageId=us-2016 GRAPHIC: Race to the White House tmsnrt.rs/1Tr5O3e '
     'Tales from the Trail blog here Reuters election newsletter here',
     'The vote is in November.'),

    # ── 不应被改动的正常文本 / normal text that must survive ────────────────
    ('He sent it via email. Then left.',
     'He sent it via email. Then left.'),
    ('The photo shows a crowd. Officials said 3.5 million people attended.',
     'The photo shows a crowd. Officials said 3.5 million people attended.'),
]

TITLE_CASES = [
    (' Lunatic Glenn Beck Prepares To Move (VIDEO)',
     'Lunatic Glenn Beck Prepares To Move'),
    ('WATCH Newest Addition To Trump’s Nat Security Council [VIDEO]',
     "Newest Addition To Trump's Nat Security Council"),
    ('BREAKING: WATCH: Something happened',
     'Something happened'),
    ('Ana Navarro Points Out Differences, And It’s HILARIOUS (TWEET)',
     "Ana Navarro Points Out Differences, And It's HILARIOUS"),
    ('Boiler Room #62 – Fatal Illusions',
     'Boiler Room #62 - Fatal Illusions'),
    # 栏目标签：真假两侧都删 / section labels: removed on both sides
    ('Factbox: Trump fills top jobs for his administration',
     'Trump fills top jobs for his administration'),
    ('Exclusive: U.S. weighs new sanctions',
     'U.S. weighs new sanctions'),
    ('FLASHBACK: Obama Said Something',
     'Obama Said Something'),
    ('Trump on Twitter (Dec 28) - Global Warming',
     'Global Warming'),
    # 煽动性用词故意保留 / sensational word choice is deliberately kept
    ('BOMBSHELL: Seth Rich Sent Emails',
     'BOMBSHELL: Seth Rich Sent Emails'),
    # 真新闻标题不应被改动 / real headlines must survive unchanged
    ("'Numerous' arrests made during Trump inauguration protests: police",
     "'Numerous' arrests made during Trump inauguration protests: police"),
    ('U.S.-backed alliance says Russian jets struck its fighters in east Syria',
     'U.S.-backed alliance says Russian jets struck its fighters in east Syria'),
]


@pytest.mark.parametrize('raw, expected', TEXT_CASES)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize('raw, expected', TITLE_CASES)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_model_text_is_lowercase_alphanumeric():
    out = to_model_text(pd.Series(['Trump’s TWEET: "WOW!!!" (U.S.)']))[0]
    assert out == 'trump s tweet wow us'


def test_model_text_unifies_dotted_acronyms():
    # Reuters 的 "U.S." 和假新闻网站的 "US" 必须变成同一个词
    # Reuters' "U.S." and the fake sites' "US" must become the same token
    out = to_model_text(pd.Series(['U.S.-backed forces', 'US-backed forces']))
    assert out[0] == out[1] == 'us backed forces'


# ─────────────────────────────────────────────────────────────────────────────
# 分组与划分 / Grouping and splitting
# ─────────────────────────────────────────────────────────────────────────────

from src import data as data_module  # noqa: E402

BODY = ' '.join(f'w{i}' for i in range(40))           # 40 个词的正文 / a 40-word body
OTHER = ' '.join(f'x{i}' for i in range(40))


def test_near_duplicate_grouping_is_transitive():
    # A 和 B 标题相同；B 和 C 正文开头相同；D 与其他都无关 → 分组应为 {A, B, C} 和 {D}
    # A~B share a title, B~C share a lead, D is unrelated → groups {A,B,C} and {D}
    df = pd.DataFrame({
        'title': ['Same five word long title', 'Same five word long title',
                  'Completely different headline here ok', 'Unrelated other headline about things'],
        'text': ['a1 ' + OTHER, BODY, BODY + ' extra ending words', 'z ' + OTHER[::-1]],
    })
    g = data_module._group_near_duplicates(df)
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]


def test_grouped_split_never_separates_a_group():
    # 构造 200 组、每组 3 篇的数据，检查每组只落在一个集合里，且比例约为 64/16/20
    # 200 groups of 3 docs: each group must land in exactly one split, ~64/16/20
    df = pd.DataFrame({'label': [i % 2 for i in range(200) for _ in range(3)],
                       'group_id': [i for i in range(200) for _ in range(3)]})
    split = data_module._assign_splits(df, seed=0)
    assert (df.assign(split=split).groupby('group_id')['split'].nunique() == 1).all()
    shares = split.value_counts(normalize=True)
    assert abs(shares['test'] - 0.20) < 0.03
    assert abs(shares['val'] - 0.16) < 0.03


def test_split_lock_detects_changed_split(tmp_path, monkeypatch):
    # 第一次运行保存划分；划分被改动后必须报错
    # The first call saves the split; a changed split must raise
    monkeypatch.setattr(data_module, 'SPLITS_PATH', tmp_path / 'splits.csv')
    monkeypatch.setattr(data_module, 'PROCESSED_DIR', tmp_path)
    df = pd.DataFrame({'doc_id': ['a', 'b', 'c'], 'split': ['train', 'val', 'test']})
    data_module._check_splits_locked(df)            # 保存 / saves
    data_module._check_splits_locked(df)            # 相同则通过 / identical → passes
    changed = df.assign(split=['train', 'test', 'val'])
    with pytest.raises(RuntimeError):
        data_module._check_splits_locked(changed)
