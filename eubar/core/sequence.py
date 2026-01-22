"""Sequence/window helpers for the refactored pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

try:
    from pyfaidx import Fasta  # type: ignore
except Exception:  # pragma: no cover
    Fasta = None  # pyfaidx is optional; only needed for genome FASTA reads


def reverse_complement(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]


def parse_snv(s: str) -> Tuple[str, int, str, str]:
    """Parse "chr:pos:ref>alt"."""
    chrom, pos_s, change = s.split(":")
    ref, alt = change.split(">")
    return chrom, int(pos_s), ref.upper(), alt.upper()


def fetch_sequence(genome_fasta: str, chrom: str, start_1based: int, end_1based_inclusive: int) -> str:
    """Fetch sequence from FASTA, 1-based inclusive coordinates."""
    if Fasta is None:
        raise ImportError(
            "pyfaidx is required to read the genome FASTA (pip install pyfaidx), "
            "or run in an environment where pyfaidx is available."
        )
    genome = Fasta(genome_fasta)
    return genome[chrom][start_1based - 1 : end_1based_inclusive].seq.upper()


@dataclass(frozen=True)
class RegionWindow:
    chrom: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    seq: str
    reverse: bool = False

    @classmethod
    def from_region_string(
        cls,
        region: str,
        genome_fasta: str,
        *,
        reverse: bool = False,
    ) -> "RegionWindow":
        chrom, coords = region.split(":")
        start_s, end_s = coords.split("-")
        start, end = int(start_s), int(end_s)
        seq = fetch_sequence(genome_fasta, chrom, start, end)
        if reverse:
            seq = reverse_complement(seq)
        return cls(chrom=chrom, start=start, end=end, seq=seq, reverse=reverse)


@dataclass(frozen=True)
class SnvWindow:
    """2*k-1 window centered on an SNV."""

    chrom: str
    pos: int
    ref: str
    alt: str
    k: int
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    seq: str
    snv_index: int  # 0-based index within seq
    flipped: bool   # whether we reverse-complemented

    @classmethod
    def from_snv(
        cls,
        snv: str,
        genome_fasta: str,
        *,
        k: int = 8,
        debug: bool = False,
    ) -> "SnvWindow":
        chrom, pos, ref, alt = parse_snv(snv)
        half = k - 1
        start = pos - half
        end = pos + half  # inclusive
        seq = fetch_sequence(genome_fasta, chrom, start, end)
        snv_index = pos - start
        base = seq[snv_index]
        flipped = False

        if base != ref:
            # try reverse complement
            seq_rc = reverse_complement(seq)
            snv_index_rc = len(seq_rc) - 1 - snv_index
            base_rc = seq_rc[snv_index_rc]
            if base_rc != ref:
                raise ValueError(
                    f"Reference allele mismatch even after reverse complement: genome={base}, rc={base_rc}, ref={ref}"
                )
            seq = seq_rc
            snv_index = snv_index_rc
            flipped = True

        if debug:
            import sys

            sys.stderr.write(
                f"[SNVWIN] {snv} start={start} end={end} snv_index={snv_index} flipped={flipped} seq={seq}\n"
            )

        return cls(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            k=k,
            start=start,
            end=end,
            seq=seq,
            snv_index=snv_index,
            flipped=flipped,
        )

    def iter_overlapping_windows(self) -> Iterator[Tuple[int, int]]:
        """Yield (motif_pos, snv_index_in_kmer) for the k windows overlapping the SNV."""
        # windows start positions such that motif_pos <= snv_index < motif_pos+k
        for motif_pos in range(0, self.k):
            snv_index_in_kmer = self.snv_index - motif_pos
            if 0 <= snv_index_in_kmer < self.k:
                yield motif_pos, snv_index_in_kmer
