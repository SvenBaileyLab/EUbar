"""Explicit spaced/gapped mask support for EUbar SNV mode.

A mask is a string such as ``11111000011111`` where:
  * 1 = informative base used for sequence matching
  * 0 = ignored spacer base

The production matcher deliberately keeps the regression layer unchanged.  It
builds a *targeted* masked-signature index for the SNVs in the current command,
then reuses that index for every SNV.  This avoids constructing the enormous
all-signatures index that a long spaced mask would otherwise require.

Probe region identifiers are assumed to use the same BED-style coordinates as
``eubar array`` (``chr:start-end``, start 0-based, end exclusive).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from .matching import MatchResults
from .rand import RandSample, deterministic_hash
from .sequence import SnvWindow, _open_fasta
from .patterns import SequenceMask as SequenceMask


_BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_CODE_TO_BASE = "ACGT"
_ASCII_TO_CODE = np.full(256, 255, dtype=np.uint8)
for _b, _c in _BASE_TO_CODE.items():
    _ASCII_TO_CODE[ord(_b)] = _c
    _ASCII_TO_CODE[ord(_b.lower())] = _c


@dataclass(frozen=True)
class MaskHit:
    offset_1based: int
    is_reverse: bool


@dataclass(frozen=True)
class MaskBuildStats:
    n_probe_regions: int
    n_valid_regions: int
    n_windows: int
    n_target_signatures: int
    n_target_region_hits: int
    n_target_ambiguous_regions: int
    n_rand_signatures: int
    rand_signature_modulus: int
    rand_reps_per_signature: int


class MaskedMotifMatcher:
    """SNV matcher backed by a targeted masked-signature probe index."""

    is_masked = True

    def __init__(
        self,
        *,
        mask: SequenceMask,
        signature_hits: Mapping[int, Mapping[str, MaskHit]],
        rand_universe: Mapping[int, Sequence[Tuple[str, int]]],
        stats: MaskBuildStats,
    ):
        self.mask = mask
        self.mask_pattern = mask.pattern
        self.signature_hits = signature_hits
        self.rand_universe = rand_universe
        self.stats = stats
        # Compatibility attribute. Mask mode uses make_rand_sampler() instead.
        self.kmer_positions: Mapping[str, Mapping[str, int]] = {}

    @classmethod
    def build(
        cls,
        *,
        mask: SequenceMask,
        snv_windows: Sequence[SnvWindow],
        probe_regions: Iterable[str],
        genome_fasta: str,
        rand_target_signatures: int = 100_000,
        rand_target_occurrences: int = 200_000,
    ) -> "MaskedMotifMatcher":
        """Build a targeted index for all candidate allele signatures.

        AFF keeps all unambiguous probe-region hits for signatures needed by the
        supplied SNVs. RAND uses a bounded, deterministic subsample of the wider
        masked-signature universe so memory does not explode for masks with many
        informative positions.
        """

        target_codes: Set[int] = set()
        for snv in snv_windows:
            for motif_pos, snv_index in snv.iter_overlapping_windows():
                if snv_index not in mask.informative:
                    continue
                context = snv.seq[motif_pos : motif_pos + mask.span]
                if len(context) != mask.span:
                    continue
                for allele in "ACGT":
                    filled = list(context)
                    filled[snv_index] = allele
                    code = mask.signature_code("".join(filled))
                    if code is not None:
                        target_codes.add(code)

        if not target_codes:
            raise ValueError("no valid masked signatures could be built for the supplied SNVs")

        # Keep RAND memory bounded. Hash-subsampling signatures preserves the
        # important family-level sampling idea without storing every occurrence.
        theoretical = 4 ** mask.n_informative
        rand_target_signatures = max(1, int(rand_target_signatures))
        modulus = max(1, int(math.ceil(theoretical / rand_target_signatures)))
        expected_selected = max(1, theoretical // modulus)
        reps_per_signature = max(
            1,
            min(32, int(rand_target_occurrences) // expected_selected),
        )

        target_hits: Dict[int, Dict[str, MaskHit]] = {
            code: {} for code in target_codes
        }
        target_ambiguous: Dict[int, Set[str]] = {
            code: set() for code in target_codes
        }
        rand_universe: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

        fasta = _open_fasta(genome_fasta)
        n_probe_regions = 0
        n_valid_regions = 0
        n_windows = 0

        for region in probe_regions:
            n_probe_regions += 1
            try:
                chrom, coords = str(region).split(":", 1)
                start_s, end_s = coords.split("-", 1)
                start0, end0 = int(start_s), int(end_s)
                seq = fasta[chrom][start0:end0].seq.upper()
            except Exception:
                continue

            if len(seq) < mask.span:
                continue
            n_valid_regions += 1

            f_codes, r_codes, valid_f, valid_r = _signature_codes(seq, mask)
            n = len(f_codes)
            n_windows += int(n)

            # Track occurrences *within this region* first so repeated masked
            # signatures can be excluded, matching the current unique-hit array
            # semantics rather than arbitrarily choosing one offset.
            local_target: Dict[int, List[MaskHit]] = defaultdict(list)
            local_rand_count: Dict[int, int] = defaultdict(int)
            local_rand_first: Dict[int, int] = {}

            for off0 in range(n):
                if bool(valid_f[off0]):
                    fc = int(f_codes[off0])
                    if fc in target_codes:
                        local_target[fc].append(
                            MaskHit(offset_1based=off0 + 1, is_reverse=False)
                        )

                    if _rand_signature_selected(fc, modulus):
                        local_rand_count[fc] += 1
                        local_rand_first.setdefault(fc, off0 + 1)

                if bool(valid_r[off0]):
                    rc = int(r_codes[off0])
                    if rc in target_codes:
                        local_target[rc].append(
                            MaskHit(offset_1based=off0 + 1, is_reverse=True)
                        )

            for code, hits in local_target.items():
                # Deduplicate exact duplicate hit tuples, then require one unique
                # occurrence/orientation for this signature in this probe region.
                uniq = {(h.offset_1based, h.is_reverse): h for h in hits}
                if len(uniq) == 1:
                    if region not in target_ambiguous[code]:
                        target_hits[code][region] = next(iter(uniq.values()))
                else:
                    target_ambiguous[code].add(region)
                    target_hits[code].pop(region, None)

            for code, count in local_rand_count.items():
                if count != 1:
                    continue
                bucket = rand_universe[code]
                if len(bucket) < reps_per_signature:
                    bucket.append((str(region), int(local_rand_first[code])))

        n_target_region_hits = sum(len(v) for v in target_hits.values())
        n_target_ambiguous = sum(len(v) for v in target_ambiguous.values())

        # Remove empty RAND buckets.
        rand_universe = {k: tuple(v) for k, v in rand_universe.items() if v}
        if not rand_universe:
            raise ValueError(
                "mask RAND universe is empty; try a mask with fewer informative positions "
                "or check probe/genome coordinates"
            )

        stats = MaskBuildStats(
            n_probe_regions=n_probe_regions,
            n_valid_regions=n_valid_regions,
            n_windows=n_windows,
            n_target_signatures=len(target_codes),
            n_target_region_hits=n_target_region_hits,
            n_target_ambiguous_regions=n_target_ambiguous,
            n_rand_signatures=len(rand_universe),
            rand_signature_modulus=modulus,
            rand_reps_per_signature=reps_per_signature,
        )
        return cls(
            mask=mask,
            signature_hits=target_hits,
            rand_universe=rand_universe,
            stats=stats,
        )

    def filter_windows(self, windows):
        return [
            (int(motif_pos), int(snv_index))
            for motif_pos, snv_index in windows
            if int(snv_index) in self.mask.informative
        ]

    def scan_snv(self, region_seq: str, k: int, overlapping_windows) -> MatchResults:
        if int(k) != self.mask.span:
            raise ValueError(
                f"masked matcher span is {self.mask.span}, but analysis requested k={k}"
            )

        allele_region_offsets: MutableMapping = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        allele_matched_kmers: MutableMapping = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for motif_pos, snv_index in self.filter_windows(overlapping_windows):
            context = region_seq[motif_pos : motif_pos + self.mask.span]
            if len(context) != self.mask.span:
                continue
            wildcard = self.mask.display_pattern(context, snv_index)

            for allele in "ACGT":
                filled = list(context)
                filled[snv_index] = allele
                filled_seq = "".join(filled)
                code = self.mask.signature_code(filled_seq)
                if code is None:
                    continue
                hits = self.signature_hits.get(code)
                if not hits:
                    continue

                dest = allele_region_offsets[motif_pos][snv_index][allele]
                out_list = allele_matched_kmers[motif_pos][snv_index][allele]
                filled_display = self.mask.filled_display(context, snv_index, allele)

                for region, hit in hits.items():
                    if hit.is_reverse:
                        wildcard_pos = (
                            int(hit.offset_1based)
                            + (self.mask.span - 1 - int(snv_index))
                        )
                    else:
                        wildcard_pos = int(hit.offset_1based) + int(snv_index)
                    dest[region] = int(wildcard_pos)
                    out_list.append((wildcard, filled_display, region))

        return MatchResults(
            allele_region_offsets=allele_region_offsets,
            allele_matched_kmers=allele_matched_kmers,
        )

    def make_rand_sampler(self) -> "MaskedRandSampler":
        return MaskedRandSampler(self.rand_universe)


class MaskedRandSampler:
    """RAND sampler over masked signature families.

    The universe contains a bounded deterministic set of masked signatures, with
    one or more unique probe-region representatives per signature.  Sampling is
    deterministic per SNV and returns *wildcard positions directly*.
    """

    def __init__(self, universe: Mapping[int, Sequence[Tuple[str, int]]]):
        self.universe = universe
        self._keys = tuple(universe.keys())
        if not self._keys:
            raise ValueError("empty masked RAND universe")

    def sample(
        self,
        *,
        matched_regions: Set[str],
        snv_str: str,
        k: int,
        rand_n: int = 500,
        positions: Optional[Mapping[int, int]] = None,
    ) -> RandSample:
        if not positions:
            positions = {j: j for j in range(int(k))}

        rng = random.Random(deterministic_hash(snv_str) + 1337)
        by_pos: Dict[int, Dict[str, int]] = {int(j): {} for j in positions}
        used_all: Set[str] = set()

        for motif_pos, snv_index in positions.items():
            motif_pos = int(motif_pos)
            snv_index = int(snv_index)
            used_j: Set[str] = set()
            attempts = 0
            max_attempts = max(10000, int(rand_n) * 300)

            while len(used_j) < int(rand_n) and attempts < max_attempts:
                attempts += 1
                sig = rng.choice(self._keys)
                reps = self.universe.get(sig)
                if not reps:
                    continue
                region, offset_1based = rng.choice(reps)
                if region in matched_regions or region in used_all or region in used_j:
                    continue

                wildcard_pos = int(offset_1based) + snv_index
                by_pos[motif_pos][region] = wildcard_pos
                used_j.add(region)
                used_all.add(region)

        return RandSample(
            by_pos=by_pos,
            used_all=used_all,
            values_are_wildcard_pos=True,
        )


def _rand_signature_selected(code: int, modulus: int) -> bool:
    if modulus <= 1:
        return True
    # 64-bit multiplicative mix; deterministic and much cheaper than hashing
    # a string for every probe window.
    x = (int(code) * 11400714819323198485 + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    return (x % int(modulus)) == 0


def _signature_codes(seq: str, mask: SequenceMask):
    """Return forward/reverse signature codes + orientation-specific validity."""
    raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    if len(raw) < mask.span:
        empty_u = np.empty(0, dtype=np.uint64)
        empty_b = np.empty(0, dtype=bool)
        return empty_u, empty_u.copy(), empty_b, empty_b.copy()
    base = _ASCII_TO_CODE[raw]
    n = len(base) - mask.span + 1
    f = np.zeros(n, dtype=np.uint64)
    r = np.zeros(n, dtype=np.uint64)
    valid_f = np.ones(n, dtype=bool)
    valid_r = np.ones(n, dtype=bool)

    for p, rp in zip(mask.informative, mask.reverse_informative):
        v = base[p : p + n]
        valid_f &= v < 4
        f = (f << np.uint64(2)) | v.astype(np.uint64)

        rv = base[rp : rp + n]
        valid_r &= rv < 4
        comp = (3 - rv).astype(np.uint64)
        r = (r << np.uint64(2)) | comp

    return f, r, valid_f, valid_r

