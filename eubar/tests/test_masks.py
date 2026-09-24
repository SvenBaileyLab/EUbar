from __future__ import annotations

import numpy as np

from eubar.core.masking import SequenceMask, _signature_codes
from eubar.core.sequence import reverse_complement


def test_arbitrary_asymmetric_mask_is_supported():
    mask = SequenceMask.parse("1100111111")
    assert mask.span == 10
    assert mask.n_informative == 8
    assert mask.informative == (0, 1, 4, 5, 6, 7, 8, 9)
    assert not mask.is_contiguous

    seq = "ACGTACGTAA"
    assert mask.display_pattern(seq, 4) == "ACxx.CGTAA"
    assert mask.filled_display(seq, 4, "T") == "ACxxTCGTAA"


def test_reverse_signature_for_asymmetric_mask_matches_reverse_complement():
    mask = SequenceMask.parse("1100111111")
    seq = "ACGTACGTAA"
    f, r, valid_f, valid_r = _signature_codes(seq, mask)

    assert len(f) == len(r) == 1
    assert bool(valid_f[0]) and bool(valid_r[0])
    assert int(f[0]) == mask.signature_code(seq)
    assert int(r[0]) == mask.signature_code(reverse_complement(seq))
