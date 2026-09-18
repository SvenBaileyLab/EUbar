"""Stable, process-independent sampling for --max-probes."""
import hashlib
import json
import numpy as np


def subsample_probes(lookup, max_probes, seed=0, context=()):
    if max_probes is None:
        return lookup
    if max_probes <= 0:
        raise ValueError("max_probes must be positive")
    total = sum(len(v) for v in lookup.values())
    if total <= max_probes:
        return lookup
    payload = json.dumps([int(seed), list(context)], separators=(",", ":"))
    stable_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable_seed)
    result = {}
    for allele, regions in sorted(lookup.items()):
        items = sorted(regions.items())
        # Preserve the existing proportional allocation, including rounding.
        n_keep = min(len(items), max(1, round(max_probes * len(items) / total)))
        selected = sorted(rng.choice(len(items), size=n_keep, replace=False))
        result[allele] = dict(items[i] for i in selected)
    return result
