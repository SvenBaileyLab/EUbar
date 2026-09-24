import math

import pytest

from eubar.snv import _holm_adjust, _apply_holm_within_snv, _best_supported_motif_pos, _best_pval_summary_rows, _format_pval


def test_holm_ignores_missing_tests_without_mutating_rows():
    rows = [
        {'label':'AFF','allele':'C','motif_pos':0,'pval':0.01},
        {'label':'AFF','allele':'C','motif_pos':1,'pval':math.nan},
        {'label':'AFF','allele':'G','motif_pos':2,'pval':0.001},
        {'label':'RAND','allele':'A','motif_pos':0,'pval':0.02},
        {'label':'RAND','allele':'C','motif_pos':0,'pval':0.03},
    ]
    got = _apply_holm_within_snv(rows, 'chr1:10:A>C')
    assert got[0]['pval_holm'] == .01
    assert math.isnan(got[1]['pval_holm'])
    assert math.isnan(got[2]['pval_holm'])
    assert got[3]['pval_holm'] == pytest.approx(.04)
    assert got[4]['pval_holm'] == pytest.approx(.04)
    assert all('pval_holm' not in r for r in rows)
    assert got[0]['pval'] == .01


def test_best_position_ties_and_unsupported_fallback():
    rows = [{'label':'AFF','allele':'C','motif_pos':p,'pval':.01,'coef':c}
            for p,c in [(4,1),(3,-2),(2,2)]]
    assert _best_supported_motif_pos(rows,'chr1:10:A>C') == 2
    summary = _best_pval_summary_rows(rows,'chr1:10:A>C')
    assert summary[0]['effect'] == 2
    assert math.isnan(summary[1]['effect'])
    assert math.isnan(summary[2]['effect'])


def test_holm_support_uses_adjusted_pvalue():
    rows = [
        {'label':'AFF','allele':'C','motif_pos':0,'pval':.001,'coef':1},
        {'label':'AFF','allele':'C','motif_pos':1,'pval':.01,'coef':1},
        {'label':'RAND','allele':'C','motif_pos':0,'pval':.03,'pval_holm':.06,'coef':1},
        {'label':'RAND','allele':'A','motif_pos':1,'pval':.01,'pval_holm':.02,'coef':1},
    ]
    assert _best_supported_motif_pos(rows,'chr1:10:A>C') == 0
    assert _best_supported_motif_pos(rows,'chr1:10:A>C',use_holm=True) == 1


def test_empty_and_underflow_output():
    assert _holm_adjust([]) == []
    assert math.isnan(_holm_adjust([math.nan])[0])
    assert _format_pval(math.nan) == 'nan'
    assert _format_pval(0) == '<1e-300'
    assert _format_pval(0,40.0) == '7.31e-350'
    summary = _best_pval_summary_rows([], 'chr1:10:A>C')
    assert [r['type'] for r in summary] == ['AFF','RAND','RAND']
    assert all(r['motif_pos'] is None for r in summary)
