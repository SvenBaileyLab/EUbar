"""K-mer matching (shared by scan and snv).

This is scan-style: SNV mode should filter scan matches, not re-implement
matching logic.

Output contract:
    allele_region_offsets[motif_pos][snv_index][allele][region_id] = wildcard_pos
    allele_matched_kmers[motif_pos][snv_index][allele] = list of (wildcard_kmer, filled_kmer, region_id)

wildcard_pos is computed so lp is strand-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import DefaultDict, Dict, List, Mapping, Tuple

from .sequence import reverse_complement


def wildcard_pos_from_offset(offset: int, j: int, kmer_size: int, is_reverse: bool = False) -> int:
    """Convert k-mer start offset to wildcard base position within the probe region."""
    if is_reverse:
        return int(offset) + (kmer_size - 1 - int(j))
    return int(offset) + int(j)


@dataclass(frozen=True)
class MatchResults:
    allele_region_offsets: Mapping[int, Mapping[int, Mapping[str, Mapping[str, int]]]]
    allele_matched_kmers: Mapping[int, Mapping[int, Mapping[str, List[Tuple[str, str, str]]]]]


class MotifMatcher:
    """Scan-style k-mer matcher."""

    def __init__(self, kmer_positions: Mapping[str, Mapping[str, int]]):
        self.kmer_positions = kmer_positions

    def scan(self, region_seq: str, k: int) -> MatchResults:
        allele_region_offsets: DefaultDict[int, DefaultDict[int, DefaultDict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        allele_matched_kmers: DefaultDict[int, DefaultDict[int, DefaultDict[str, List[Tuple[str, str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        seen = set()  # prevent duplicate matches per motif_pos + snv + base + region + match_kmer

        for motif_pos in range(len(region_seq) - k + 1):
            kmer = region_seq[motif_pos : motif_pos + k]

            for snv_index in range(k):
                wildcard = list(kmer)
                wildcard[snv_index] = "."
                wildcard_kmer = "".join(wildcard)

                for base in "ACGT":
                    filled = list(kmer)
                    filled[snv_index] = base
                    filled_kmer = "".join(filled)
                    rc = reverse_complement(filled_kmer)

                    for match_kmer in (filled_kmer, rc):
                        region_map = self.kmer_positions.get(match_kmer)
                        if not region_map:
                            continue
                        is_reverse = match_kmer == rc
                        for region_id, offset in region_map.items():
                            key = (motif_pos, snv_index, base, region_id, match_kmer)
                            if key in seen:
                                continue
                            seen.add(key)

                            wp = wildcard_pos_from_offset(
                                offset=int(offset),
                                j=snv_index,
                                kmer_size=k,
                                is_reverse=is_reverse,
                            )
                            allele_region_offsets[motif_pos][snv_index][base][region_id] = wp
                            allele_matched_kmers[motif_pos][snv_index][base].append(
                                (wildcard_kmer, filled_kmer, region_id)
                            )

        return MatchResults(allele_region_offsets=allele_region_offsets, allele_matched_kmers=allele_matched_kmers)

    def scan_snv(self, region_seq: str, k: int, overlapping_windows) -> MatchResults:
        """Fast SNV-only scan.

        The full :meth:`scan` enumerates *every* wildcard position (snv_index) for
        each k-mer window. For SNV mode we only need the single wildcard position
        that corresponds to the SNV for each overlapping window.

        Parameters
        ----------
        region_seq:
            The 2*k-1 SNV window sequence.
        k:
            k-mer size.
        overlapping_windows:
            Iterable of (motif_pos, snv_index_in_kmer) pairs, typically
            ``snv.iter_overlapping_windows()``.

        Returns
        -------
        MatchResults
            Same contract as :meth:`scan`, but only populated for the requested
            windows.
        """

        allele_region_offsets: DefaultDict[int, DefaultDict[int, DefaultDict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        allele_matched_kmers: DefaultDict[int, DefaultDict[int, DefaultDict[str, List[Tuple[str, str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for motif_pos, snv_index in overlapping_windows:
            motif_pos = int(motif_pos)
            snv_index = int(snv_index)
            kmer = region_seq[motif_pos : motif_pos + k]
            if len(kmer) != k:
                continue

            wildcard = list(kmer)
            wildcard[snv_index] = "."
            wildcard_kmer = "".join(wildcard)

            for base in "ACGT":
                filled = list(kmer)
                filled[snv_index] = base
                filled_kmer = "".join(filled)
                rc = reverse_complement(filled_kmer)

                # Avoid double-work for palindromic kmers
                match_kmers = (filled_kmer,) if rc == filled_kmer else (filled_kmer, rc)

                for match_kmer in match_kmers:
                    region_map = self.kmer_positions.get(match_kmer)
                    if not region_map:
                        continue
                    is_reverse = match_kmer == rc

                    dest = allele_region_offsets[motif_pos][snv_index][base]
                    out_list = allele_matched_kmers[motif_pos][snv_index][base]
                    for region_id, offset in region_map.items():
                        # If the same region is hit via both strands/palindrome, keep
                        # the first assignment (they should imply the same wildcard_pos
                        # for unique-hit indices anyway).
                        if region_id in dest:
                            continue

                        wp = wildcard_pos_from_offset(
                            offset=int(offset),
                            j=snv_index,
                            kmer_size=k,
                            is_reverse=is_reverse,
                        )
                        dest[region_id] = wp
                        out_list.append((wildcard_kmer, filled_kmer, region_id))

        return MatchResults(allele_region_offsets=allele_region_offsets, allele_matched_kmers=allele_matched_kmers)
