"""Guard the tolerance used for cross-machine reference comparisons."""
import pytest

from output_comparison import assert_table_equal


def test_last_digit_rounding_is_allowed():
    assert_table_equal('effect\tpval\n0.123000000000001\t7.310000000000001e-350\n',
                       'effect\tpval\n0.123\t7.31e-350\n')
    assert_table_equal('chr1:0-23\t-0.8948422014036953\n',
                       'chr1:0-23\t-0.8948422014036952\n', header=False)


@pytest.mark.parametrize('actual, expected', [
    ('effect\n0.124\n', 'effect\n0.123\n'),
    ('pval\n7.31e-349\n', 'pval\n7.31e-350\n'),
    ('pval\n0\n', 'pval\n7.31e-350\n'),
    ('effect\n1e-30\n', 'effect\n0\n'),
    ('effect\n0\n', 'effect\nNA\n'),
    ('effect\nnan\n', 'effect\nNA\n'),
    ('n_probes\n3000000000001\n', 'n_probes\n3000000000000\n'),
    ('F\n50.00000000000001\n', 'F\n50\n'),
    ('mask\n01101\n', 'mask\n1101\n'),
    ('recommended_mask\n11010\n', 'recommended_mask\n11011\n'),
    ('allele\teffect\nT\t0.1\nA\t0.2\n', 'allele\teffect\nA\t0.2\nT\t0.1\n'),
    ('effect\n0.1\n', 'effect\n0.1\n0.2\n'),
    ('coef\n0.1\n', 'effect\n0.1\n'),
    ('effect\n0.1\t0.2\n', 'effect\n0.1\n'),
    ('effect\n0.1', 'effect\n0.1\n'),
])
def test_scientific_or_structural_changes_are_rejected(actual, expected):
    with pytest.raises(AssertionError):
        assert_table_equal(actual, expected)
