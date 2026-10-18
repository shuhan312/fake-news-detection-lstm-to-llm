"""
证据检索的单元测试（不联网）/ Tests for evidence retrieval (no network).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evidence  # noqa: E402


def test_entity_spans_split_on_full_stop_and_drop_filler():
    spans = evidence._entity_spans('Says Iowa Gov. Terry Branstad chartered a plane')
    assert 'Terry Branstad' in spans and 'Iowa' in spans
    assert not any(s.startswith('Says') for s in spans)


def test_build_queries_drop_numbers_and_noise():
    q = evidence.build_queries('Says 21,000 Wisconsin residents got jobs in 2011, but 18,000 of them were in other states.')
    assert q['content'].split()[:4] == ['Wisconsin', 'residents', 'got', 'jobs']
    assert not any(ch.isdigit() for ch in q['content'])
    assert 'Says' not in q['content']


def test_factcheck_paragraph_is_leaky():
    para = 'PolitiFact rated the claim Pants on Fire, noting that the governor never chartered a plane.'
    assert evidence.is_leaky(para, 'Says the governor chartered a plane') == 'factcheck'


def test_restated_claim_with_verdict_is_leaky_but_plain_background_is_not():
    claim = 'Obama used the Koran instead of the Bible when sworn into office'
    leaky = 'Some chain emails falsely claimed that Obama was sworn into office using the Koran instead of the Bible.'
    background = 'Obama was sworn into office on January 20, 2009, using the Bible owned by Abraham Lincoln.'
    assert evidence.is_leaky(leaky, claim) == 'restated_verdict'
    assert evidence.is_leaky(background, claim) == ''        # 真实背景资料必须保留 / genuine background is kept


def test_select_passages_ranks_relevant_paragraph_first_and_counts_removed():
    statement = 'Rick Perry has advocated abandoning Social Security'
    pages = {'Political positions of Rick Perry': '\n'.join([
        'Perry has called Social Security a Ponzi scheme and suggested that states should be able to opt out of the federal program entirely in his book.',
        'Perry was born in Paint Creek, Texas, and served in the United States Air Force as a pilot flying cargo planes in the Middle East and Europe.',
        # 至少 20 个词，才会进入泄漏检查（更短的段落在长度过滤时就被跳过）
        # 20+ words so it reaches the leak check (shorter paragraphs are skipped earlier)
        'PolitiFact rated a related claim about Perry and Social Security as half true after carefully reviewing his book and several of his speeches.',
    ])}
    passages, removed = evidence.select_passages(statement, pages)
    assert 'Ponzi' in passages[0]['text']
    assert removed['factcheck'] == 1


def test_parse_queries_keeps_two_nonempty_strings():
    assert evidence.parse_queries('{"queries": ["US employment rate", "", "1970s labor force", "extra"]}') \
        == ['US employment rate', '1970s labor force']
    assert evidence.parse_queries('not json') == []
    assert evidence.parse_queries('{"queries": "single string"}') == []


def test_reference_list_entries_are_not_used_as_evidence():
    pages = {'Forecasting': '\n'.join([
        'Rescher, Nicholas (1998). Predicting the Future: An Introduction to the Theory of Forecasting. State University of New York Press. ISBN 978-0-7914-3553-3.',
        'State revenue forecasts are produced each year by budget offices and are frequently revised when tax collections differ from what was projected in the budget.',
    ])}
    passages, _ = evidence.select_passages('State revenue projections have missed the mark', pages)
    assert all('ISBN' not in p['text'] for p in passages)
    assert passages and passages[0]['text'].startswith('State revenue forecasts')


def test_articles_cataloguing_false_claims_are_not_used():
    # 来自测试集证据审计：这类文章的标题本身就说明"这些是假的"
    # From the test-evidence audit: the title itself says "these are false"
    text = ('Trump repeatedly claimed that The Apprentice was the top-rated television program in America, '
            'which television ratings published at the time did not support in any season of the show.')
    passages, removed = evidence.select_passages('Trump said The Apprentice was the top-rated show',
                                                 {'False or misleading statements by Donald Trump': text})
    assert passages == [] and removed['verdict_article'] == 1


def test_author_date_citation_of_a_fact_check_is_not_used():
    # 来自测试集证据审计：《华盛顿邮报》核查文章的标题直接泄漏了结论
    # From the test-evidence audit: a Washington Post fact-check headline leaks the verdict
    citation = ('Kessler, Glenn (January 21, 2013). "The stale claim that 40 percent of gun sales lack '
                'background checks". The Washington Post (blog). Retrieved from the archive later.')
    background = ('A gun show loophole refers to private sales at gun shows, which in many states do not '
                  'require the seller to run a background check on the buyer under federal law.')
    passages, removed = evidence.select_passages('40 percent of gun sales happen without a background check',
                                                 {'Gun show loophole': citation + '\n' + background})
    assert all('stale claim' not in p['text'] for p in passages)
    assert removed['reference'] == 1


def test_articles_about_fact_checkers_are_dropped_by_title():
    # val 审计中发现：标题为 "PolitiFact" 的文章被检索到 / found in the val audit: an article titled "PolitiFact"
    from src.evidence import select_passages
    para = ' '.join(['The 2022 Lie of the Year was disinformation about the invasion of Ukraine'] * 3)
    passages, removed = select_passages('Says Mark Warner told the Lie of the Year in 2013.',
                                        {'PolitiFact': para, 'Mark Warner': para.replace('Lie of the Year', 'senator')})
    assert all(p['title'] != 'PolitiFact' for p in passages)
    assert removed['verdict_article'] == 1
