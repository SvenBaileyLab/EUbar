from __future__ import annotations

from eubar.core.matching import MotifMatcher
from eubar.core.sequence import reverse_complement


def _plain(mapping):
    """Recursively turn defaultdict-ish mappings into ordinary dicts."""
    if hasattr(mapping, "items"):
        return {k: _plain(v) for k, v in mapping.items()}
    if isinstance(mapping, list):
        return list(mapping)
    return mapping


def test_fast_snv_scan_matches_full_scan_for_requested_windows():
    seq = "AACGTAA"
    k = 4
    center = k - 1
    windows = [(motif_pos, center - motif_pos) for motif_pos in range(k)]

    # Build an index containing every allele substitution that the requested
    # SNV windows can query. Each kmer gets a unique region so comparisons are clear.
    index = {}
    counter = 0
    for motif_pos, snv_index in windows:
        local = seq[motif_pos : motif_pos + k]
        for base in "ACGT":
            filled = list(local)
            filled[snv_index] = base
            kmer = "".join(filled)
            for key in {kmer, reverse_complement(kmer)}:
                counter += 1
                index.setdefault(key, {})[f"chr1:{counter*10}-{counter*10+20}"] = 2

    matcher = MotifMatcher(index)
    full = matcher.scan(seq, k)
    fast = matcher.scan_snv(seq, k, windows)

    for motif_pos, snv_index in windows:
        assert _plain(fast.allele_region_offsets[motif_pos][snv_index]) == _plain(
            full.allele_region_offsets[motif_pos][snv_index]
        )
        assert _plain(fast.allele_matched_kmers[motif_pos][snv_index]) == _plain(
            full.allele_matched_kmers[motif_pos][snv_index]
        )
