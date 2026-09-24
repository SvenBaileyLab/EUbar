"""Compare baseline tables without requiring identical floating-point rounding."""
from decimal import Decimal, InvalidOperation

# Only measured quantities may vary. IDs, counts, positions, masks, flags and
# recommendations remain exact, even when they happen to look like numbers.
NUMERIC_COLUMNS = set('''
A C G T coef effect pval raw_pval E E_reduced p IC
eligible_fraction minor_allele_fraction minor_allele_fraction_median
residual_sd_ratio residual_sd_ratio_median residual_sd_ratio_p90
success_fraction full_effect full_effect_abs_norm full_effect_abs_norm_median
full_neglog10p neglog10p_median neglog10p_p90
effect_sd_norm effect_sd_norm_median effect_sd_norm_p90
effect_bias_norm effect_bias_norm_median effect_bias_norm_p90
effect_rmse_norm effect_rmse_norm_median
stability_ratio_to_best stability_pct_above_best
stability_ratio_to_best_candidate stability_pct_above_best_candidate
near_optimal_threshold_effect_sd_norm_p90
best_control_effect_sd_norm_p90 best_control_pct_vs_recommended
random_control_effect_sd_norm_median random_control_effect_sd_norm_p10
random_control_effect_sd_norm_p90
recommended_eligible_fraction recommended_effect_sd_norm_p90
recommended_effect_bias_norm_p90 previous_eligible_fraction
previous_effect_sd_norm_p90 previous_effect_bias_norm_p90
previous_effect_sd_excess previous_effect_bias_excess previous_eligible_deficit
'''.split())
RELATIVE_TOLERANCE = Decimal('1e-12')


def assert_table_equal(actual, expected, *, header=True):
    """Require exact table structure and text; allow 1e-12 relative numeric drift.

    There is no absolute tolerance: zero and tiny p-values stay meaningful.
    Decimal also preserves p-values smaller than the range of a Python float.
    Headerless intensity tables contain an interval ID and one measured value.
    """
    if actual == expected:
        return
    actual_lines = actual.splitlines(keepends=True)
    expected_lines = expected.splitlines(keepends=True)
    assert len(actual_lines) == len(expected_lines), 'table row count changed'
    if header:
        assert actual_lines[0] == expected_lines[0], 'table header changed'
        columns = expected_lines[0].rstrip('\r\n').split('\t')
    else:
        columns = ['interval', 'intensity']
    for row, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines), 1):
        a = actual_line.rstrip('\r\n')
        e = expected_line.rstrip('\r\n')
        assert actual_line[len(a):] == expected_line[len(e):], f'row {row}: newline changed'
        ac, ec = a.split('\t'), e.split('\t')
        assert len(ac) == len(ec), f'row {row}: column count changed'
        for col, (value, reference) in enumerate(zip(ac, ec)):
            if value == reference:
                continue
            label = columns[col] if col < len(columns) else ''
            location = f'row {row}, {label}: {value!r} != {reference!r}'
            assert label in NUMERIC_COLUMNS or (not header and col == 1), location
            assert value.strip() == value and reference.strip() == reference, location
            try:
                observed, baseline = Decimal(value), Decimal(reference)
            except InvalidOperation:
                raise AssertionError(location) from None
            assert observed.is_finite() and baseline.is_finite(), location
            assert abs(observed - baseline) <= RELATIVE_TOLERANCE * abs(baseline), location


def assert_output_equal(actual_path, expected_path):
    if expected_path.suffix == '.meme':
        assert actual_path.read_bytes() == expected_path.read_bytes(), expected_path.name
    else:
        assert_table_equal(actual_path.read_text(), expected_path.read_text(),
                           header=expected_path.name != 'intensities.tsv')
