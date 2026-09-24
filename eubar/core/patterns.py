"""Sequence span and informative positions, independent of probe indexing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

_BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass(frozen=True)
class SequenceMask:
    pattern: str
    informative: Tuple[int, ...]

    @classmethod
    def parse(cls, pattern: str) -> "SequenceMask":
        p = str(pattern).strip()
        if not p:
            raise ValueError("mask cannot be empty")
        if set(p) - {"0", "1"}:
            raise ValueError("mask must contain only 0 and 1")
        informative = tuple(i for i, ch in enumerate(p) if ch == "1")
        if len(informative) < 2:
            raise ValueError("mask must contain at least two informative (1) positions")
        if len(informative) > 31:
            # Signatures are encoded in a uint64 (2 bits/base).
            raise ValueError("mask currently supports at most 31 informative positions")
        return cls(pattern=p, informative=informative)

    @classmethod
    def from_options(cls, kmer_size: int = 8, mask: Optional[str] = None) -> "SequenceMask":
        """Represent either mode without routing contiguous matching through masks."""
        if mask is not None:
            return cls.parse(mask)
        # The 31-base encoding limit applies to explicit masks, not legacy k-mers.
        return cls("1" * kmer_size, tuple(range(kmer_size)))

    @property
    def reverse_informative(self) -> Tuple[int, ...]:
        """Positions visited by reverse-complement signature encoding, in order."""
        return tuple(self.span - 1 - p for p in self.informative)

    @property
    def span(self) -> int:
        return len(self.pattern)

    @property
    def n_informative(self) -> int:
        return len(self.informative)

    @property
    def is_contiguous(self) -> bool:
        return all(ch == "1" for ch in self.pattern)

    def signature_code(self, seq: str) -> Optional[int]:
        if len(seq) != self.span:
            return None
        code = 0
        for pos in self.informative:
            base = seq[pos].upper()
            val = _BASE_TO_CODE.get(base)
            if val is None:
                return None
            code = (code << 2) | val
        return int(code)

    def display_pattern(self, seq: str, snv_index: int) -> str:
        """Human-readable masked context: x=ignored, .=variant position."""
        if len(seq) != self.span:
            return ""
        out: List[str] = []
        for i, base in enumerate(seq):
            if i == snv_index:
                out.append(".")
            elif self.pattern[i] == "0":
                out.append("x")
            else:
                out.append(base.upper())
        return "".join(out)

    def filled_display(self, seq: str, snv_index: int, allele: str) -> str:
        if len(seq) != self.span:
            return ""
        out: List[str] = []
        for i, base in enumerate(seq):
            if self.pattern[i] == "0":
                out.append("x")
            elif i == snv_index:
                out.append(allele.upper())
            else:
                out.append(base.upper())
        return "".join(out)
