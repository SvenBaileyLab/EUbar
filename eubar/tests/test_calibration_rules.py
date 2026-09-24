from __future__ import annotations

import pandas as pd
import pytest

from eubar.core.calibration import (
    build_mask_recommendation,
    generate_matched_random_masks,
    summarize,
)


def test_max_probe_rule_is_strict_but_marks_borderline_previous_cap():
    family_info = pd.DataFrame({"n_probes_available": [1500] * 10})
    family_curves = pd.DataFrame(
        [
            # cap 1000 barely misses SD threshold
            *[
                {"cap_probes": 1000, "effect_sd_norm": 0.1004, "effect_bias_norm": 0.01,
                 "effect_rmse_norm": 0.05, "neglog10p_median": 1.0, "n_probes_kept_median": 1000}
                for _ in range(10)
            ],
            # cap 1250 passes
            *[
                {"cap_probes": 1250, "effect_sd_norm": 0.08, "effect_bias_norm": 0.01,
                 "effect_rmse_norm": 0.04, "neglog10p_median": 1.0, "n_probes_kept_median": 1250}
                for _ in range(10)
            ],
        ]
    )

    _, rec = summarize(
        family_curves,
        family_info,
        [1000, 1250],
        max_effect_sd=0.10,
        max_effect_bias=0.10,
        min_eligible_fraction=0.50,
        borderline_relative_tolerance=0.05,
    )
    row = rec.iloc[0]
    assert row["recommended_max_probes"] == 1250
    assert bool(row["previous_cap_borderline"])
    assert row["previous_tested_cap"] == 1000


def test_random_mask_controls_preserve_span_information_content_and_seed():
    ref = "1111100000011111"  # span 16, ten informative bases
    a = generate_matched_random_masks(ref, 12, seed=7)
    b = generate_matched_random_masks(ref, 12, seed=7)
    assert a == b
    assert len(a) == len(set(a)) == 12
    for mask in a:
        assert len(mask) == len(ref)
        assert mask.count("1") == ref.count("1")
        assert mask[0] == mask[-1] == "1"
        assert mask != ref


def test_mask_recommendation_never_selects_control_and_reports_control_failure():
    summary = pd.DataFrame(
        [
            {"mask": "11111000011111", "mask_role": "candidate", "control_source": "candidate",
             "eligible_fraction": 0.90, "n_families_evaluated": 500,
             "effect_sd_norm_p90": 0.30, "effect_bias_norm_p90": 0.04,
             "residual_sd_ratio_p90": 0.98, "minor_allele_fraction_median": 0.14},
            {"mask": "11011011011011", "mask_role": "control", "control_source": "random",
             "eligible_fraction": 0.91, "n_families_evaluated": 500,
             "effect_sd_norm_p90": 0.27, "effect_bias_norm_p90": 0.04,
             "residual_sd_ratio_p90": 0.98, "minor_allele_fraction_median": 0.15},
        ]
    )

    annotated, rec = build_mask_recommendation(
        summary,
        stability_probes=100,
        min_eligible_fraction=0.5,
        near_optimal_relative_tolerance=0.05,
        random_control_reference_mask="11111000011111",
        n_random_controls_requested=1,
    )
    row = rec.iloc[0]
    assert row["recommended_mask"] == "11111000011111"
    assert row["control_status"] == "CONTROL_OUTPERFORMS_RECOMMENDED_CANDIDATE"
    assert row["random_control_candidate_percentile"] == pytest.approx(0.0)
    assert row["random_control_fraction_better_or_equal"] == pytest.approx(1.0)

    control = annotated.loc[annotated["mask_role"] == "control"].iloc[0]
    assert bool(control["beats_recommended_candidate"])
