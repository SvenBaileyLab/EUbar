from __future__ import annotations

import math

import pytest

from eubar.snv import (
    _apply_holm_within_snv,
    _best_supported_motif_pos,
    _holm_adjust,
)


def test_holm_adjust_known_example():
    # Sorted p-values: .01, .03, .04 -> Holm: .03, .06, .06
    got = _holm_adjust([0.01, 0.04, 0.03])
    assert got == pytest.approx([0.03, 0.06, 0.06])


def test_holm_families_are_aff_and_rand_separately():
    snv = "chr1:100:A>C"
    rows = []
    # Ten AFF tests -> the smallest raw p gets multiplied by 10.
    for j in range(10):
        rows.append({"label": "AFF", "allele": "C", "motif_pos": j, "pval": 0.001 * (j + 1)})
    # Twenty RAND tests: REF + ALT over ten positions -> smallest multiplied by 20.
    for j in range(10):
        rows.append({"label": "RAND", "allele": "A", "motif_pos": j, "pval": 0.0001 * (j + 1)})
        rows.append({"label": "RAND", "allele": "C", "motif_pos": j, "pval": 0.0002 * (j + 1)})

    adjusted = _apply_holm_within_snv(rows, snv)
    aff0 = next(r for r in adjusted if r["label"] == "AFF" and r["motif_pos"] == 0)
    rand0 = next(r for r in adjusted if r["label"] == "RAND" and r["allele"] == "A" and r["motif_pos"] == 0)

    assert aff0["pval_holm"] == pytest.approx(0.01)
    assert rand0["pval_holm"] == pytest.approx(0.002)


def test_best_pval_skips_better_aff_without_rand_support():
    snv = "chr1:100:A>C"
    rows = [
        # Motif position 0 has the best AFF p-value but no positive significant RAND support.
        {"label": "AFF", "allele": "C", "motif_pos": 0, "pval": 1e-6, "coef": 1.5},
        {"label": "RAND", "allele": "A", "motif_pos": 0, "pval": 0.50, "coef": 0.2},
        {"label": "RAND", "allele": "C", "motif_pos": 0, "pval": 0.01, "coef": -0.2},
        # Position 1 has a weaker AFF p-value but supported positive RAND effect.
        {"label": "AFF", "allele": "C", "motif_pos": 1, "pval": 1e-4, "coef": 1.0},
        {"label": "RAND", "allele": "A", "motif_pos": 1, "pval": 0.01, "coef": 0.3},
        {"label": "RAND", "allele": "C", "motif_pos": 1, "pval": 0.40, "coef": 0.1},
    ]
    assert _best_supported_motif_pos(rows, snv, use_holm=False) == 1
