from __future__ import annotations

from eubar.core.rand import RandSampler


def test_rand_sampling_is_deterministic_even_after_cache_warmup():
    kmers = {}
    for i in range(30):
        k = f"K{i:02d}"
        kmers[k] = {f"chr1:{i*100+j*10}-{i*100+j*10+20}": j + 1 for j in range(6)}

    sampler = RandSampler(kmers)
    kwargs = dict(matched_regions=set(), snv_str="chr1:100:A>C", k=4, rand_n=5)
    first = sampler.sample(**kwargs)
    # Call with another SNV to mutate/warm the internal region-key cache.
    sampler.sample(matched_regions=set(), snv_str="chr1:101:A>G", k=4, rand_n=5)
    second = sampler.sample(**kwargs)

    assert first.by_pos == second.by_pos
    assert first.used_all == second.used_all
