from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eubar.core.masking import SequenceMask
from eubar.motifs import (
    _mask_pattern_from_code,
    _mask_seed_to_code,
    ppm_to_bits_matrix,
    reverse_complement_ppm,
)


def test_masked_motif_seed_roundtrip_preserves_spacer():
    mask = SequenceMask.parse("11111000011111")
    pattern = "TTGGC....GCCAA"
    code = _mask_seed_to_code(pattern, mask)
    assert _mask_pattern_from_code(mask, code) == pattern


def test_uniform_spacer_positions_have_zero_information_bits():
    ppm = pd.DataFrame(
        [
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.0, "C": 0.0, "G": 1.0, "T": 0.0},
        ]
    )
    bits = ppm_to_bits_matrix(ppm)
    assert bits.iloc[1].to_numpy().sum() == pytest.approx(0.0, abs=1e-10)
    assert bits.iloc[0].sum() == pytest.approx(2.0, abs=1e-9)
    assert bits.iloc[2].sum() == pytest.approx(2.0, abs=1e-9)


def test_reverse_complement_ppm_swaps_bases_and_reverses_positions():
    ppm = pd.DataFrame(
        [
            {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
            {"A": 0.1, "C": 0.6, "G": 0.2, "T": 0.1},
        ]
    )
    rc = reverse_complement_ppm(ppm)
    # Original final row C=.6 becomes first RC row G=.6.
    assert rc.iloc[0]["G"] == pytest.approx(0.6)
    # Original first row A=.7 becomes final RC row T=.7.
    assert rc.iloc[1]["T"] == pytest.approx(0.7)
