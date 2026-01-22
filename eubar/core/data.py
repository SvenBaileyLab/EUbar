"""Data loading and indexing for the refactored EUBAR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


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


@dataclass(frozen=True)
class KmerIndex:
    """Simple dictionary-backed index: kmer -> {region: offset}."""

    kmers: Mapping[str, Mapping[str, int]]

    @classmethod
    def from_file(cls, path: str) -> "KmerIndex":
        return cls(read_unique_kmer_positions(path))

    def lookup(self, kmer: str) -> Optional[Mapping[str, int]]:
        return self.kmers.get(kmer)
