# eubar/core/data.py
"""Data loading and indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Iterable, Tuple


def read_intensities(path: str) -> Dict[str, float]:
    """Read "region value" lines into a dict."""
    out: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            region = parts[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            out[region] = val
    return out


def read_unique_kmer_positions(path: str) -> Dict[str, Dict[str, int]]:
    """Read kmer positions (unique hits only).

    Format per line:
        kmer<TAB>region1;count1;offsets1,region2;count2;offsets2,...

    Only entries with count==1 are retained; offset is parsed as int.
    """
    result: Dict[str, Dict[str, int]] = {}
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if "\t" not in raw:
                continue
            kmer, entries = raw.split("\t", 1)
            regions: Dict[str, int] = {}
            for entry in entries.strip(",").split(","):
                if not entry:
                    continue
                parts = entry.split(";")
                if len(parts) != 3:
                    continue
                region, count_s, offsets_s = parts
                try:
                    if int(count_s) != 1:
                        continue
                    offset = int(offsets_s.strip())
                except ValueError:
                    continue
                regions[region] = offset
            if regions:
                result[kmer] = regions
    return result


@dataclass(frozen=True)
class IntensityTable:
    values: Mapping[str, float]

    @classmethod
    def from_file(cls, path: str) -> "IntensityTable":
        return cls(read_intensities(path))

    def get(self, region: str) -> Optional[float]:
        return self.values.get(region)

    # --- notebook/debug ergonomics ---
    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, region: str) -> bool:
        return region in self.values

    def keys(self) -> Iterable[str]:
        return self.values.keys()

    def items(self) -> Iterable[Tuple[str, float]]:
        return self.values.items()

    def __repr__(self) -> str:
        return f"IntensityTable(n={len(self.values)})"


@dataclass(frozen=True)
class KmerIndex:
    """Simple dictionary-backed index: kmer -> {region: offset}."""

    kmers: Mapping[str, Mapping[str, int]]

    @classmethod
    def from_file(cls, path: str) -> "KmerIndex":
        return cls(read_unique_kmer_positions(path))

    def lookup(self, kmer: str) -> Optional[Mapping[str, int]]:
        return self.kmers.get(kmer)

    # --- notebook/debug ergonomics ---
    def __len__(self) -> int:
        return len(self.kmers)

    def __contains__(self, kmer: str) -> bool:
        return kmer in self.kmers

    def keys(self) -> Iterable[str]:
        return self.kmers.keys()

    def items(self) -> Iterable[Tuple[str, Mapping[str, int]]]:
        return self.kmers.items()

    def __repr__(self) -> str:
        return f"KmerIndex(n_kmers={len(self.kmers)})"
